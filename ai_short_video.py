#!/usr/bin/env python3
# ai_short_video.py
# Python 3.11+
#
# End-to-end short video generator:
#  - generate a short script via OpenAI GPT
#  - synthesize audio via ElevenLabs (preferred) or gTTS fallback
#  - generate images with OpenAI Images (DALL·E) or call a Stable Diffusion endpoint
#  - assemble into MP4 with MoviePy
#
# Replace placeholders with your API keys or set them in a .env file.

import os
import time
import json
import logging
import random
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import requests
from requests import Response
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
from pydub import AudioSegment  # for format conversion if needed
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# --------------------------
# Configuration / placeholders
# --------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY_HERE")  # optional
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "YOUR_STABILITY_API_KEY_HERE")  # optional (if using SD cloud)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_IMAGES_URL = "https://api.openai.com/v1/images/generations"  # DALL·E-ish endpoint
ELEVENLABS_VOICE_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
RETRY_BASE = 1.2
MAX_RETRIES = 3

# --------------------------
# Utilities
# --------------------------
def safe_request(method: str, url: str, **kwargs) -> Response:
    """Simple wrapper with retry and basic error handling."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.debug("Request: %s %s (attempt %d)", method, url, attempt)
            resp = requests.request(method, url, timeout=30, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504):
                logging.warning("Transient error %s on %s, attempt %d", resp.status_code, url, attempt)
                time.sleep(RETRY_BASE ** attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logging.warning("Request exception on attempt %d: %s", attempt, exc)
            if attempt == MAX_RETRIES:
                logging.error("Max retries reached for %s", url)
                raise
            time.sleep(RETRY_BASE ** attempt)
    raise RuntimeError("Unreachable safe_request flow")  # defensive

# --------------------------
# 1) Script / Story generation (OpenAI GPT)
# --------------------------
def generate_script_openai(
    topic: str,
    length_seconds_target: int = 30,
    model: str = "gpt-4o-mini",  # example; change to gpt-4, gpt-4o, gpt-5... depending on your access
) -> str:
    """
    Generate a short, spoken script for a short-form video.
    Returns plain text script.
    """
    prompt = (
        f"Write a short, engaging spoken script for a ~{length_seconds_target}s short video about \"{topic}\".\n"
        "Requirements:\n"
        "- Keep language natural and conversational.\n"
        "- Include a brief opening hook (1-2 lines), 2-4 content sentences, and a closing call-to-action.\n"
        "- Keep total length suitable for ~30 seconds spoken at normal pace (about 60-80 words).\n"
        "- Return only the script (no analysis or metadata)."
    )
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 350,
    }

    resp = safe_request("POST", OPENAI_CHAT_COMPLETIONS_URL, headers=headers, json=body)
    data = resp.json()
    try:
        script = data["choices"][0]["message"]["content"].strip()
        logging.info("Generated script:\n%s", script)
        return script
    except Exception as e:
        logging.error("Unexpected OpenAI response structure: %s", data)
        raise RuntimeError("Failed to parse OpenAI response") from e

# --------------------------
# 2a) Text-to-speech via ElevenLabs
# --------------------------
def tts_elevenlabs(text: str, voice: str = "alloy", output_path: Path = OUTPUT_DIR / "tts_eleven.mp3") -> Path:
    """
    Use ElevenLabs TTS to produce an mp3. Requires ELEVENLABS_API_KEY.
    Replace voice id with available voice id.
    """
    if ELEVENLABS_API_KEY in (None, "", "YOUR_ELEVENLABS_API_KEY_HERE"):
        raise RuntimeError("ElevenLabs API key not set. Set ELEVENLABS_API_KEY in env or .env")

    voice_id = voice  # if you have real voice IDs, insert here; some accounts use '21m00Tcm4TlvDq8ikWAM'
    url = ELEVENLABS_VOICE_URL_TEMPLATE.format(voice_id=voice_id)
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    body = {"text": text, "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}

    # Many ElevenLabs endpoints require a POST to /v1/text-to-speech/{voice_id} with content-type JSON
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.post(url, headers=headers, json=body, stream=True, timeout=60)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info("Saved ElevenLabs TTS to %s", output_path)
            return output_path
        else:
            logging.warning("ElevenLabs TTS failed (%d): %s", resp.status_code, resp.text)
            if attempt == MAX_RETRIES:
                raise RuntimeError("ElevenLabs TTS failed after retries")
            time.sleep(RETRY_BASE ** attempt)
    raise RuntimeError("Unreachable ElevenLabs flow")

# --------------------------
# 2b) Text-to-speech via gTTS (fallback, offline-friendly)
# --------------------------
def tts_gtts(text: str, lang: str = "en", output_path: Path = OUTPUT_DIR / "tts_gtts.mp3") -> Path:
    """
    Use gTTS as a free fallback. Requires internet for gTTS to fetch Google's TTS.
    """
    try:
        from gtts import gTTS
    except Exception as exc:
        logging.error("gTTS library missing: %s", exc)
        raise

    tts = gTTS(text, lang=lang)
    tts.save(str(output_path))
    logging.info("Saved gTTS audio to %s", output_path)
    return output_path

# --------------------------
# 3) Image generation
# Provide two methods: OpenAI Images (DALL·E) and an example stable-diffusion cloud endpoint
# --------------------------
def generate_images_openai(prompt: str, n_images: int = 3, size: str = "1024x1024") -> List[Path]:
    """
    Generate images using OpenAI Images (DALL·E) endpoint.
    Returns list of local file paths.
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"prompt": prompt, "n": n_images, "size": size}
    resp = safe_request("POST", OPENAI_IMAGES_URL, headers=headers, json=body)
    data = resp.json()

    image_paths: List[Path] = []
    # OpenAI Images responses often include base64 data or URLs. Support both.
    for i, img_obj in enumerate(data.get("data", [])):
        # try url first
        img_path = OUTPUT_DIR / f"image_openai_{i+1}.png"
        if "b64_json" in img_obj:
            import base64
            b64 = img_obj["b64_json"]
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64))
            image_paths.append(img_path)
        elif "url" in img_obj:
            url = img_obj["url"]
            r = safe_request("GET", url, stream=True)
            with open(img_path, "wb") as f:
                f.write(r.content)
            image_paths.append(img_path)
        else:
            logging.warning("Unknown image object: %s", img_obj)
    logging.info("Saved %d images using OpenAI Images", len(image_paths))
    return image_paths

def generate_images_stability(prompt: str, n_images: int = 3, size: str = "1024x1024") -> List[Path]:
    """
    Example: call a Stability.ai / other Stable Diffusion HTTP API.
    This is a placeholder — modify to match your provider (Replicate, Stability, DreamStudio).
    """
    if STABILITY_API_KEY in (None, "", "YOUR_STABILITY_API_KEY_HERE"):
        raise RuntimeError("Stability API key not set. Set STABILITY_API_KEY in env or .env")

    # Example using Stability REST v1 (pseudo). Replace with actual endpoint & parameters.
    url = "https://api.stability.ai/v1/generation/text-to-image"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Content-Type": "application/json"}
    body = {"text_prompts": [{"text": prompt}], "cfg_scale": 7, "samples": n_images, "width": 1024, "height": 1024}

    resp = safe_request("POST", url, headers=headers, json=body)
    data = resp.json()
    image_paths: List[Path] = []
    for idx, item in enumerate(data.get("artifacts", [])):
        b64 = item.get("base64")
        if not b64:
            continue
        import base64
        img_path = OUTPUT_DIR / f"image_sd_{idx+1}.png"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(b64))
        image_paths.append(img_path)
    logging.info("Saved %d images via Stability endpoint", len(image_paths))
    return image_paths

# --------------------------
# 4) Video Assembly (MoviePy)
# --------------------------
@dataclass
class VideoAssemblyConfig:
    image_duration: float = 4.0  # seconds per image
    width: int = 1280
    height: int = 720
    text_overlay_font: Optional[str] = None  # system font path or None (MoviePy default)
    text_overlay_size: int = 36
    output_file: Path = OUTPUT_DIR / "final_video.mp4"

def assemble_video(
    image_paths: List[Path],
    audio_path: Path,
    script_text: str,
    cfg: VideoAssemblyConfig,
) -> Path:
    """
    Create a simple slideshow video:
      - each image shown for cfg.image_duration
      - audio is the TTS track
      - overlay script as subtitles (static at bottom) - simple implementation
    Returns path to generated MP4.
    """
    logging.info("Assembling video with %d images and audio %s", len(image_paths), audio_path)
    # Load audio duration
    audio_clip = AudioFileClip(str(audio_path))
    audio_duration = audio_clip.duration
    logging.info("Audio duration: %.2fs", audio_duration)

    # Determine total duration based either on audio or images count
    total_image_time = len(image_paths) * cfg.image_duration
    total_duration = max(audio_duration, total_image_time)
    logging.info("Total video duration target: %.2fs", total_duration)

    # Create ImageClips
    image_clips = []
    for idx, img in enumerate(image_paths):
        clip = ImageClip(str(img)).set_duration(cfg.image_duration)
        clip = clip.resize(width=cfg.width)
        # add optional text overlay per clip or static overlay
        # we'll add a static bottom text with script (wrapped)
        image_clips.append(clip)

    # If audio longer than images total, stretch last image duration to fill audio
    if audio_duration > total_image_time and image_clips:
        extra = audio_duration - total_image_time
        logging.info("Extending last image by %.2fs to match audio", extra)
        image_clips[-1] = image_clips[-1].set_duration(cfg.image_duration + extra)

    # Concatenate
    video = concatenate_videoclips(image_clips, method="compose")
    # Add text overlay (simple)
    # MoviePy TextClip may require ImageMagick or fallback. Keep it simple: center bottom text.
    subtitle = TextClip(txt=script_text, fontsize=cfg.text_overlay_size, font=cfg.text_overlay_font or "Arial", method="caption", size=(cfg.width - 100, None))
    subtitle = subtitle.set_position(("center", cfg.height - 120)).set_duration(video.duration)

    final = CompositeVideoClip([video, subtitle.set_start(0)])
    final = final.set_audio(audio_clip)
    final = final.set_duration(video.duration)

    # Write file
    out = cfg.output_file
    logging.info("Writing final video to %s ...", out)
    final.write_videofile(
        str(out),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
        verbose=False,
        progress_bar=True,
    )
    logging.info("Video exported: %s", out)
    return out

# --------------------------
# 5) Helper to break script into image prompts (simple heuristics)
# --------------------------
def create_image_prompts_from_script(script: str, n_images: int = 3) -> List[str]:
    """
    Turn script into N image prompts by splitting sentences and adding style hints.
    This is simple heuristic; you can refine with a call to an LLM for richer prompts.
    """
    # split into sentences naively
    candidates = [s.strip() for s in script.replace("\n", " ").split(".") if s.strip()]
    prompts = []
    for i in range(n_images):
        idx = i if i < len(candidates) else random.randrange(len(candidates))
        base = candidates[idx] if candidates else script
        prompt = f"{base}. Cinematic, high-detail, vibrant colors, 16:9 framing"
        prompts.append(prompt)
    logging.info("Generated image prompts: %s", prompts)
    return prompts

# --------------------------
# 6) CLI / main orchestration
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="AI short video generator")
    parser.add_argument("--topic", type=str, required=True, help="Topic or trending topic for the short video")
    parser.add_argument("--voice", type=str, default="alloy", help="ElevenLabs voice id or name (if using ElevenLabs)")
    parser.add_argument("--use_eleven", action="store_true", help="Use ElevenLabs for TTS (default fallback: gTTS)")
    parser.add_argument("--image_provider", choices=["openai", "stability"], default="openai")
    parser.add_argument("--n_images", type=int, default=3)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "final_video.mp4"))
    args = parser.parse_args()

    # Step 1: generate script
    script = generate_script_openai(topic=args.topic, length_seconds_target=30)

    # Step 2: TTS
    audio_path = OUTPUT_DIR / "tts.mp3"
    try:
        if args.use_eleven:
            audio_path = tts_elevenlabs(script, voice=args.voice, output_path=audio_path)
        else:
            audio_path = tts_gtts(script, output_path=audio_path)
    except Exception as e:
        logging.error("TTS failed: %s", e)
        raise

    # Step 3: Generate image prompts and images
    prompts = create_image_prompts_from_script(script, n_images=args.n_images)
    image_paths: List[Path] = []
    try:
        if args.image_provider == "openai":
            # synthesize images in a loop to keep resource usage friendly
            # join prompts to fetch images per prompt
            for i, p in enumerate(prompts):
                imgs = generate_images_openai(prompt=p, n_images=1, size="1280x720")
                if imgs:
                    image_paths.extend(imgs)
        else:
            for p in prompts:
                imgs = generate_images_stability(prompt=p, n_images=1, size="1024x1024")
                if imgs:
                    image_paths.extend(imgs)
    except Exception as e:
        logging.error("Image generation failed: %s", e)
        raise

    if not image_paths:
        raise RuntimeError("No images produced; aborting")

    # Step 4: assemble video
    cfg = VideoAssemblyConfig(
        image_duration= max(3.0, 30.0 / max(1, len(image_paths))),  # naive per-image duration
        width=1280,
        height=720,
        text_overlay_font=None,
        text_overlay_size=36,
        output_file=Path(args.output),
    )
    final_path = assemble_video(image_paths=image_paths, audio_path=audio_path, script_text=script, cfg=cfg)

    logging.info("Done. Final video at: %s", final_path)

if __name__ == "__main__":
    main()
