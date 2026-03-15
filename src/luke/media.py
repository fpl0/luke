"""Media processing: image encoding, video frames, whisper transcription, prompt building."""

from __future__ import annotations

import asyncio
import base64
import io
import re
from pathlib import Path
from typing import Any

import structlog
from structlog.stdlib import BoundLogger

from . import db
from .config import settings

log: BoundLogger = structlog.get_logger()

_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
_MAX_IMAGE_DIM = 1568  # Anthropic Vision API max dimension


# ---------------------------------------------------------------------------
# Media processing
# ---------------------------------------------------------------------------


async def transcribe(path: Path) -> str | None:
    """Transcribe an audio file using Whisper. Returns text or None."""
    import mlx_whisper

    try:
        result: dict[str, Any] = await asyncio.wait_for(
            asyncio.to_thread(mlx_whisper.transcribe, str(path), path_or_hf_repo=_WHISPER_MODEL),
            timeout=settings.transcription_timeout,
        )
        text: str = result.get("text", "").strip()
        if not text:
            return None
        path.with_suffix(".txt").write_text(text, encoding="utf-8")
        return text
    except TimeoutError:
        log.error("whisper_timeout", path=str(path), timeout=settings.transcription_timeout)
        return None
    except Exception:
        log.exception("whisper_transcription_failed", path=str(path))
        return None


async def extract_frame(video: Path, output: Path) -> bool:
    """Extract first frame from video using ffmpeg. Returns True on success."""
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            str(video),
            "-vframes",
            "1",
            "-f",
            "image2",
            "-y",
            str(output),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await asyncio.wait_for(proc.wait(), timeout=settings.ffmpeg_timeout) == 0
    except TimeoutError:
        log.error("ffmpeg_timeout", video=str(video), timeout=settings.ffmpeg_timeout)
        if proc is not None:
            proc.kill()
        return False
    except FileNotFoundError:
        log.debug("ffmpeg_not_found")
        return False
    except Exception:
        log.exception("frame_extraction_failed", video=str(video))
        return False


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

_MEDIA_RE = re.compile(r"\[(Photo|Sticker image|Animation frame|Video thumbnail) saved: (.+?)\]")

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def encode_image(path: Path) -> dict[str, Any] | None:
    """Read, resize, and base64-encode an image file using Pillow."""
    import contextlib

    import PIL.Image
    from PIL import ImageOps

    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size == 0:
        return None

    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    try:
        with PIL.Image.open(path) as img:
            # Fix EXIF rotation (e.g. portrait phone photos)
            with contextlib.suppress(Exception):
                img = ImageOps.exif_transpose(img) or img
            img.thumbnail(
                (_MAX_IMAGE_DIM, _MAX_IMAGE_DIM),
                PIL.Image.Resampling.LANCZOS,
            )
            buf = io.BytesIO()
            fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
            out_img = img.convert("RGB") if fmt == "JPEG" else img
            if fmt == "JPEG":
                media_type = "image/jpeg"
            out_img.save(buf, format=fmt, quality=85)
        data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    except Exception:
        log.exception("image_encode_failed", path=str(path))
        return None


# ---------------------------------------------------------------------------
# Multimodal prompt building
# ---------------------------------------------------------------------------


async def build_prompt(
    messages: list[db.StoredMessage],
    chat_id: str = "",
) -> str | list[dict[str, Any]]:
    """Build a text or multimodal prompt from pending messages."""

    def _fmt(m: db.StoredMessage) -> str:
        reply_ctx = ""
        if m.reply_to and chat_id:
            ref = db.get_message_by_msg_id(chat_id, m.reply_to)
            if ref:
                reply_ctx = (
                    f" replying-to:msg:{m.reply_to}"
                    f" (original from {ref['sender']}: {ref['content'][:200]})"
                )
            else:
                reply_ctx = f" replying-to:msg:{m.reply_to}"
        return f"[{m.sender_name} {m.timestamp} msg:{m.message_id}{reply_ctx}] {m.content}"

    text_lines = [_fmt(m) for m in messages]
    full_text = "\n".join(text_lines)

    # Quick check: any media markers at all?
    all_matches = list(_MEDIA_RE.finditer(full_text))
    if not all_matches:
        return full_text

    # Build multimodal content blocks
    blocks: list[dict[str, Any]] = []
    image_count = 0

    # Enforce limit — keep most recent images
    if settings.max_images_per_prompt <= 0:
        return full_text
    skip_indices = set(range(max(0, len(all_matches) - settings.max_images_per_prompt)))

    pos = 0
    for match_idx, match in enumerate(all_matches):
        # Add text before this match
        if match.start() > pos:
            blocks.append({"type": "text", "text": full_text[pos : match.start()]})

        media_path = Path(match.group(2))

        if match_idx not in skip_indices:
            image_block = await asyncio.to_thread(encode_image, media_path)
            if image_block:
                blocks.append(image_block)
                image_count += 1
                pos = match.end()
                continue
            # Image encoding failed — annotate so the agent knows
            blocks.append(
                {
                    "type": "text",
                    "text": match.group(0) + " [image not viewable — encoding failed]",
                }
            )
            pos = match.end()
            continue

        # Fallback: keep the text marker (skipped due to batch limit)
        blocks.append({"type": "text", "text": match.group(0)})
        pos = match.end()

    # Add any remaining text
    if pos < len(full_text):
        blocks.append({"type": "text", "text": full_text[pos:]})

    if image_count == 0:
        return full_text

    # Merge adjacent text blocks
    merged: list[dict[str, Any]] = []
    for block in blocks:
        if block["type"] == "text" and merged and merged[-1]["type"] == "text":
            merged[-1]["text"] += block["text"]
        else:
            merged.append(block)

    return merged
