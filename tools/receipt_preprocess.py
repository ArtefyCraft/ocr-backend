"""
Receipt photo preprocessing — paper-band crop + vertical tiling under the
Anthropic Vision ~1.15MP server-side downscale budget.

Marker: _RECEIPT_TILING_v1 (plan lovely-sprouting-hinton F2, 2026-08-30)

Why: every corpus photo exceeds ~1.15MP, so Anthropic downscales ALL of them
to ~804x1429 regardless of source size. On long receipts the paper occupies
only 32-58% of the frame width, leaving glyphs at ~8px effective — below
reliable digit discrimination (~15-20px). Cropping to the paper band and
tiling tall crops keeps every tile under the budget so NOTHING gets
downscaled, roughly doubling effective text size.

Design constraints:
- Pure PIL (Pillow>=10 already in requirements.txt; no OpenCV in the image).
- NEVER raises to the caller: any failure returns the original image as a
  single tile with fallback=True, so the n8n bridge behaves byte-identically
  to today when preprocessing cannot help.
- Env kill-switch RECEIPT_PREPROCESS=off returns passthrough immediately.
"""
from __future__ import annotations

import base64
import io
import math
import os
from typing import Any

try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

# Anthropic downscales anything over ~1.15MP; stay under with margin.
TILE_PIXEL_BUDGET = 1_100_000
TILE_OVERLAP = 0.18          # 15-20% per plan — never cut a text line in half
# F2a (critic slice 2026-08-30): tiling DISABLED — crop-only ships first.
# Multi-image dedup semantics + uniform-overlap tiler land separately in F2b.
TILING_ENABLED = False
MIN_BAND_FRAC = 0.18         # paper band narrower than this => detection failed
MAX_BAND_FRAC = 0.97         # band ~whole frame => nothing to crop
MIN_BAND_INK_FRAC = 0.02     # candidate band must contain ink (dark px) — P0-3 guard
CROP_MARGIN_FRAC = 0.02      # keep a little context around the paper
JPEG_QUALITY = 95            # subsampling=0 below — protect 8-16px glyphs (critic P1)
MAX_INPUT_MEGAPIXELS = 24    # bigger inputs pass through untouched (memory guard)
ANALYSIS_WIDTH = 200         # downsample width for the projection profiles


def _paper_band(grey: "Image.Image") -> tuple[int, int] | None:
    """Locate the bright vertical paper band via column brightness projection.

    Returns (x0, x1) in the grey image's coordinates, or None when detection
    is unreliable (band too narrow / too wide / low contrast).
    """
    w, h = grey.size
    scale = ANALYSIS_WIDTH / w if w > ANALYSIS_WIDTH else 1.0
    small = grey.resize((max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else grey
    sw, sh = small.size
    px = small.load()

    col_mean = [sum(px[x, y] for y in range(sh)) / sh for x in range(sw)]
    lo, hi = min(col_mean), max(col_mean)
    if hi - lo < 20:  # flat profile — uniform background or uniform paper
        return None
    thresh = lo + (hi - lo) * 0.55

    bright = [x for x in range(sw) if col_mean[x] >= thresh]
    if not bright:
        return None
    # largest contiguous bright run (the paper), tolerate 2-col gaps (folds)
    runs: list[tuple[int, int]] = []
    start = prev = bright[0]
    for x in bright[1:]:
        if x - prev <= 2:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    x0s, x1s = max(runs, key=lambda r: r[1] - r[0])

    frac = (x1s - x0s + 1) / sw
    if frac < MIN_BAND_FRAC or frac > MAX_BAND_FRAC:
        return None

    # P0-3 ink guard (critic): a bright band with no dark pixels is a bare
    # table, not a receipt — cropping to it would DISCARD the receipt. Require
    # the candidate band to contain ink, and more ink than what it discards.
    dark_thresh = lo + (hi - lo) * 0.35
    def dark_frac(x_from: int, x_to: int) -> float:
        cols = range(max(0, x_from), min(sw, x_to))
        n = 0
        total = 0
        for x in cols:
            for y in range(sh):
                total += 1
                if px[x, y] < dark_thresh:
                    n += 1
        return (n / total) if total else 0.0
    band_ink = dark_frac(x0s, x1s + 1)
    # Measured on corpus (2026-08-30): real receipts 0.08-0.30, blank surfaces 0.00.
    # No outside-band comparison — a dark table/bedding outside the band is
    # background, not ink, and would wrongly veto every receipt on dark cloth.
    if band_ink < MIN_BAND_INK_FRAC or band_ink > 0.60:
        return None

    # Hysteresis edge extension (2026-08-31, wa_01 live finding): a skewed
    # receipt or edge shadow makes brightness fall off gradually at the paper
    # edge; the strict 55% detection threshold then clips the PRICE COLUMN at
    # the right edge (items read, amounts lost). Extend both edges while
    # columns stay above a laxer 35% threshold.
    ext_thresh = lo + (hi - lo) * 0.35
    while x0s > 0 and col_mean[x0s - 1] >= ext_thresh:
        x0s -= 1
    while x1s < sw - 1 and col_mean[x1s + 1] >= ext_thresh:
        x1s += 1

    inv = 1.0 / scale if scale < 1.0 else 1.0
    return int(x0s * inv), int(min(w, (x1s + 1) * inv))


def _tile_heights(width: int, height: int) -> list[tuple[int, int]]:
    """Split `height` into overlapping vertical windows, each under budget."""
    if width * height <= TILE_PIXEL_BUDGET:
        return [(0, height)]
    max_tile_h = max(1, TILE_PIXEL_BUDGET // width)
    step = max(1, int(max_tile_h * (1 - TILE_OVERLAP)))
    n = max(1, math.ceil((height - max_tile_h) / step) + 1)
    tiles = []
    for i in range(n):
        top = min(i * step, height - max_tile_h) if height > max_tile_h else 0
        tiles.append((top, min(height, top + max_tile_h)))
    # de-dup identical trailing windows
    out = []
    for t in tiles:
        if not out or t != out[-1]:
            out.append(t)
    return out


def preprocess_receipt_image(file_b64: str, mime_type: str | None = None) -> dict[str, Any]:
    """Crop to the paper band and tile under the Vision pixel budget.

    Returns {tiles, tile_count, fallback, crop_meta|error}. Never raises.
    """
    passthrough = {"tiles": [file_b64], "tile_count": 1, "fallback": True}

    if os.environ.get("RECEIPT_PREPROCESS", "on").lower() in ("off", "0", "false"):
        return {**passthrough, "error": "RECEIPT_PREPROCESS=off"}
    if not _HAS_PIL:
        return {**passthrough, "error": "Pillow unavailable"}
    if mime_type == "application/pdf":
        return {**passthrough, "error": "pdf passthrough"}

    try:
        raw = base64.b64decode(file_b64)
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # phone photos carry orientation EXIF
        img = img.convert("RGB")
        w, h = img.size
        if w * h > MAX_INPUT_MEGAPIXELS * 1_000_000:
            return {**passthrough, "error": f"input {w}x{h} over MP cap — passthrough"}

        grey = img.convert("L")
        band = _paper_band(grey)
        if not band:
            # detection unreliable — behave exactly as today (no crop, no re-encode)
            return {**passthrough, "error": "band detection declined — passthrough"}
        margin = int(w * CROP_MARGIN_FRAC)
        x0 = max(0, band[0] - margin)
        x1 = min(w, band[1] + margin)
        img = img.crop((x0, 0, x1, h))
        cw, ch = img.size

        # F2a: autocontrast REMOVED (critic P1 — unmeasured confound; colour-cast
        # risk on stamps/handwriting). Candidate for its own flagged A/B later.

        windows = _tile_heights(cw, ch) if TILING_ENABLED else [(0, ch)]
        tiles_b64 = []
        for top, bottom in windows:
            tile = img.crop((0, top, cw, bottom))
            buf = io.BytesIO()
            # q95 + no chroma subsampling: protect the 8-16px glyphs this whole
            # exercise exists to rescue (critic P1)
            tile.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True)
            tiles_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        return {
            "tiles": tiles_b64,
            "tile_count": len(tiles_b64),
            "fallback": False,
            "crop_meta": {
                "orig_w": w, "orig_h": h,
                "band": list(band),
                "crop_w": cw, "crop_h": ch,
                "windows": [list(t) for t in windows],
                "tiling_enabled": TILING_ENABLED,
            },
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        return {**passthrough, "error": f"{type(exc).__name__}: {exc}"}
