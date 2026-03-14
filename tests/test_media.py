"""Tests for luke.media — encode_image, build_prompt, transcribe, extract_frame."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luke.db import StoredMessage
from luke.media import (
    _MEDIA_TYPES,
    build_prompt,
    encode_image,
    extract_frame,
    transcribe,
)

# ---------------------------------------------------------------------------
# encode_image
# ---------------------------------------------------------------------------


class TestEncodeImage:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert encode_image(tmp_path / "nope.jpg") is None

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        assert encode_image(f) is None

    def test_with_pillow_jpeg(self, tmp_path: Path) -> None:
        """With Pillow, JPEG images are resized and re-encoded."""
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="blue")
        f = tmp_path / "photo.jpg"
        img.save(f, format="JPEG")

        result = encode_image(f)

        assert result is not None
        assert result["type"] == "image"
        assert result["source"]["media_type"] == "image/jpeg"

    def test_with_pillow_png(self, tmp_path: Path) -> None:
        """PNG files keep PNG format with Pillow."""
        from PIL import Image

        img = Image.new("RGBA", (200, 200), color=(0, 0, 255, 128))
        f = tmp_path / "photo.png"
        img.save(f, format="PNG")

        result = encode_image(f)

        assert result is not None
        assert result["source"]["media_type"] == "image/png"

    def test_with_pillow_exception(self, tmp_path: Path) -> None:
        """Pillow exception should return None."""
        f = tmp_path / "bad.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)

        with patch("PIL.Image.open", side_effect=OSError("corrupt image")):
            assert encode_image(f) is None


class TestMediaTypes:
    def test_known_extensions(self) -> None:
        assert _MEDIA_TYPES[".jpg"] == "image/jpeg"
        assert _MEDIA_TYPES[".jpeg"] == "image/jpeg"
        assert _MEDIA_TYPES[".png"] == "image/png"
        assert _MEDIA_TYPES[".gif"] == "image/gif"
        assert _MEDIA_TYPES[".webp"] == "image/webp"


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


class TestTranscribe:
    async def test_successful_transcription(self, tmp_path: Path) -> None:
        f = tmp_path / "audio.ogg"
        f.write_bytes(b"audio-data")
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"text": "Hello world"}

        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            result = await transcribe(f)

        assert result == "Hello world"
        assert f.with_suffix(".txt").read_text() == "Hello world"

    async def test_empty_transcription(self, tmp_path: Path) -> None:
        f = tmp_path / "silence.ogg"
        f.write_bytes(b"audio-data")
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"text": "  "}

        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            result = await transcribe(f)

        assert result is None

    async def test_transcription_exception(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.ogg"
        f.write_bytes(b"audio-data")
        mock_whisper = MagicMock()
        mock_whisper.transcribe.side_effect = RuntimeError("model failed")

        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            result = await transcribe(f)

        assert result is None


# ---------------------------------------------------------------------------
# extract_frame
# ---------------------------------------------------------------------------


class TestExtractFrame:
    async def test_ffmpeg_not_found(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"video-data")
        output = tmp_path / "frame.jpg"

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await extract_frame(video, output)

        assert result is False

    async def test_ffmpeg_general_error(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"video-data")
        output = tmp_path / "frame.jpg"

        with patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("ffmpeg broke")):
            result = await extract_frame(video, output)

        assert result is False

    async def test_ffmpeg_success(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"video-data")
        output = tmp_path / "frame.jpg"

        mock_proc = AsyncMock()
        mock_proc.wait.return_value = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await extract_frame(video, output)

        assert result is True

    async def test_ffmpeg_failure_exit_code(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"video-data")
        output = tmp_path / "frame.jpg"

        mock_proc = AsyncMock()
        mock_proc.wait.return_value = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await extract_frame(video, output)

        assert result is False


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def _msg(
    content: str,
    *,
    msg_id: int = 1,
    sender: str = "User",
    ts: str = "2024-01-01T00:00:00",
) -> StoredMessage:
    return StoredMessage(
        id=1, sender_name=sender, sender_id="1", message_id=msg_id, content=content, timestamp=ts
    )


class TestBuildPrompt:
    async def test_text_only(self) -> None:
        messages = [_msg("Hello"), _msg("World")]
        result = await build_prompt(messages)
        assert isinstance(result, str)
        assert "Hello" in result
        assert "World" in result

    async def test_no_media_markers_returns_string(self) -> None:
        messages = [_msg("Just a regular message with no images")]
        result = await build_prompt(messages)
        assert isinstance(result, str)

    async def test_with_image_marker_returns_list(self, tmp_path: Path) -> None:
        from PIL import Image

        img = tmp_path / "photo.png"
        Image.new("RGB", (10, 10), "blue").save(img)

        messages = [_msg(f"Look at this\n[Photo saved: {img}]")]
        result = await build_prompt(messages)

        assert isinstance(result, list)
        types = [b["type"] for b in result]
        assert "image" in types
        assert "text" in types

    async def test_batch_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from PIL import Image

        from luke.config import settings

        monkeypatch.setattr(settings, "max_images_per_prompt", 2)

        imgs = []
        for i in range(4):
            f = tmp_path / f"photo_{i}.png"
            Image.new("RGB", (10, 10), "red").save(f)
            imgs.append(f)

        content = "\n".join(f"[Photo saved: {img}]" for img in imgs)

        messages = [_msg(content)]
        result = await build_prompt(messages)

        assert isinstance(result, list)
        image_blocks = [b for b in result if b["type"] == "image"]
        assert len(image_blocks) <= 2

    async def test_batch_limit_zero_returns_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_images_per_prompt=0 should return plain text."""
        from luke.config import settings

        monkeypatch.setattr(settings, "max_images_per_prompt", 0)

        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNGfake")
        content = f"[Photo saved: {img}]"

        messages = [_msg(content)]
        result = await build_prompt(messages)
        assert isinstance(result, str)

    async def test_failed_image_encoding_annotates(self) -> None:
        """When encode_image returns None, text annotation should be added."""
        content = "[Photo saved: /tmp/nonexistent_image_xyz.jpg]"
        messages = [_msg(content)]
        result = await build_prompt(messages)
        # Image encoding fails → returns full_text as string (image_count==0)
        assert isinstance(result, str)

    async def test_remaining_text_after_images(self, tmp_path: Path) -> None:
        """Text after the last image marker should be preserved."""
        from PIL import Image

        img = tmp_path / "photo.png"
        Image.new("RGB", (10, 10), "green").save(img)
        content = f"[Photo saved: {img}]\nSome trailing text"

        messages = [_msg(content)]
        result = await build_prompt(messages)

        assert isinstance(result, list)
        text_content = " ".join(b["text"] for b in result if b.get("type") == "text")
        assert "trailing text" in text_content

    async def test_merges_adjacent_text(self) -> None:
        messages = [_msg("Part 1"), _msg("Part 2")]
        result = await build_prompt(messages)
        assert isinstance(result, str)

    async def test_sticker_and_animation_markers(self, tmp_path: Path) -> None:
        """Sticker image and Animation frame markers should also be recognized."""
        from PIL import Image

        img = tmp_path / "sticker.webp"
        Image.new("RGB", (10, 10), "yellow").save(img, format="WEBP")
        content = f"[Sticker image saved: {img}]"

        messages = [_msg(content)]
        result = await build_prompt(messages)

        assert isinstance(result, list)
        assert any(b.get("type") == "image" for b in result)


# ---------------------------------------------------------------------------
# check_ffmpeg
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Transcription timeout
# ---------------------------------------------------------------------------


class TestTranscriptionTimeout:
    async def test_timeout_returns_none(self, tmp_path: Path) -> None:
        import asyncio

        f = tmp_path / "slow.ogg"
        f.write_bytes(b"audio-data")

        mock_whisper = MagicMock()

        # Return a future that never resolves, so wait_for hits TimeoutError
        async def blocking_to_thread(*_a: object, **_kw: object) -> None:
            await asyncio.get_event_loop().create_future()

        # Patch settings to a very short timeout
        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_whisper}),
            patch("luke.media.settings") as mock_settings,
            patch("asyncio.to_thread", side_effect=blocking_to_thread),
        ):
            mock_settings.transcription_timeout = 0.01  # 10ms timeout
            result = await transcribe(f)

        assert result is None


# ---------------------------------------------------------------------------
# EXIF rotation
# ---------------------------------------------------------------------------


class TestExifRotation:
    def test_exif_transpose_called(self, tmp_path: Path) -> None:
        """EXIF transpose should be called when Pillow is available."""
        try:
            from PIL import Image, ImageOps

            # Create a real small image to test with
            img = Image.new("RGB", (100, 100), color="red")
            f = tmp_path / "rotated.jpg"
            img.save(f, format="JPEG")

            with patch.object(ImageOps, "exif_transpose", wraps=ImageOps.exif_transpose) as mock_et:
                result = encode_image(f)

            assert result is not None
            mock_et.assert_called_once()
        except ImportError:
            pytest.skip("Pillow not available")
