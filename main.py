"""Radvinex transparent 600x600 product images, with a 10% logo watermark.

Dependencies: automatically installed at first run if missing (internet required).
AI restoration: Real-ESRGAN x4 with automatically downloaded weights.
Run: python radvinex_product_png.py

Folder layout (next to this script):
    products/      source product photos
    logo.png       preferably transparent logo
    output/        generated transparent PNGs
"""

from __future__ import annotations

import gc
import time
import warnings
import importlib.util
import subprocess
import sys
from pathlib import Path

# On the first run, install required packages using this same Python interpreter.
# Internet access and installation permissions are necessary. AI packages can be
# large, especially PyTorch. If installation fails, report the error explicitly.
AUTO_INSTALL_DEPENDENCIES = True
_DEPENDENCY_GROUPS = (
    (("PIL",), ("pillow",)),
    (("rembg",), ("rembg",)),
    (("onnxruntime",), ("onnxruntime",)),
    (("torch", "torchvision"), ("torch", "torchvision")),
    (("cv2",), ("opencv-python",)),
    (("basicsr", "realesrgan"), ("basicsr", "realesrgan")),
)


def ensure_dependencies():
    if not AUTO_INSTALL_DEPENDENCIES:
        return
    for modules, packages in _DEPENDENCY_GROUPS:
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        if not missing:
            continue
        print(f"Installing missing dependencies: {', '.join(packages)}", flush=True)
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *packages]
        try:
            subprocess.run(command, check=True)
            importlib.invalidate_caches()
        except (subprocess.CalledProcessError, OSError) as exc:
            raise SystemExit(
                f"Automatic installation failed for {', '.join(packages)}. "
                f"Check internet connection and Python permissions. Error: {exc}"
            ) from exc
        still_missing = [name for name in modules if importlib.util.find_spec(name) is None]
        if still_missing:
            raise SystemExit(f"Installation finished but modules remain unavailable: {still_missing}")
    print("Required libraries are installed.", flush=True)


ensure_dependencies()

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_DIR = BASE_DIR / "products"
OUTPUT_DIR = BASE_DIR / "output"
LOGO_PATH = BASE_DIR / "logo.png"
DEBUG_DIR = BASE_DIR / "debug-masks"

CANVAS_SIZE = (600, 600)
PRODUCT_PADDING = 20                 # Product fills up to 560 x 560, ratio maintained
LOGO_OPACITY = 0.10                  # 10 percent of logo's original alpha
LOGO_COVER_CANVAS = True            # True: cover 600x600, crop excess if necessary
SKIP_EXISTING = False
SAVE_DEBUG_MASKS = False
MAX_INPUT_DIMENSION = 2400

# Same three models and mask-combination method as the supplied script
MODEL_NAMES = ("birefnet-general", "birefnet-massive", "isnet-general-use")
MODEL_LOAD_RETRIES = 3
MASK_GAMMA = 0.80
THIRD_MODEL_STRONG_THRESHOLD = 205
RECOVERY_NEIGHBORHOOD_SIZE = 21
MASK_CLOSE_SIZE = 5
MASK_EXPAND_SIZE = 3
EDGE_BLUR_RADIUS = 0.55
ALPHA_ZERO_THRESHOLD = 2
PRESERVE_PRODUCT = True

# High-quality reconstruction settings. AI super-resolution is attempted for
# small images; dependencies are installed automatically if absent. Strong Pillow enhancement
# runs automatically. AI may hallucinate tiny text/screws, so inspect catalog shots.
ENHANCE_SMALL_IMAGES = True
LOW_RESOLUTION_THRESHOLD = 1100       # Longest original source side
AI_SUPER_RESOLUTION = True
AI_MODEL_FILE = BASE_DIR / "weights" / "RealESRGAN_x4plus.pth"
AI_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
AI_AUTO_DOWNLOAD_MODEL = True
AI_RECONSTRUCTION_MAX_SIDE = 1100    # Restrict AI input for CPU/RAM
AI_USE_HALF_PRECISION = False        # Works safely without CUDA
STRONG_DENOISE = True
SHARPNESS_FACTOR = 1.42
CONTRAST_FACTOR = 1.09
UNSHARP_PERCENT = 165
UPSCALE_SMALL_CROPS = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_models():
    try:
        from rembg import new_session
    except ImportError as exc:
        raise SystemExit("Install requirements: pip install pillow rembg onnxruntime") from exc

    sessions = []
    for name in MODEL_NAMES:
        for attempt in range(1, MODEL_LOAD_RETRIES + 1):
            try:
                print(f"Loading model {name}, attempt {attempt}/{MODEL_LOAD_RETRIES} ...")
                sessions.append(new_session(name))
                break
            except Exception:
                if attempt == MODEL_LOAD_RETRIES:
                    raise
                time.sleep(5)
    return sessions


def get_ai_mask(image, session):
    from rembg import remove
    result = remove(image, session=session, only_mask=True)
    if not isinstance(result, Image.Image):
        raise ValueError("AI did not return an image mask")
    return result.convert("L")


def boost_mask(mask):
    lut = [round(255 * (x / 255) ** MASK_GAMMA) for x in range(256)]
    return mask.point(lut)


def strong_mask(mask, threshold):
    return mask.point(lambda x: 255 if x >= threshold else 0)


def build_product_mask(image, sessions, stem):
    mask1, mask2, mask3 = [get_ai_mask(image, session) for session in sessions]
    if SAVE_DEBUG_MASKS:
        for idx, mask in enumerate((mask1, mask2, mask3), 1):
            mask.save(DEBUG_DIR / f"{stem}-{idx}-mask.png")

    # Two BiRefNet predictions are combined to avoid losing fine product parts.
    primary = ImageChops.lighter(boost_mask(mask1), boost_mask(mask2))
    nearby = primary.filter(ImageFilter.MaxFilter(RECOVERY_NEIGHBORHOOD_SIZE))
    nearby = nearby.point(lambda x: 255 if x > 5 else 0)
    recovered_near = ImageChops.multiply(mask3, nearby)
    recovered_strong = strong_mask(mask3, THIRD_MODEL_STRONG_THRESHOLD)
    combined = ImageChops.lighter(primary, recovered_near)
    combined = ImageChops.lighter(combined, recovered_strong)

    if PRESERVE_PRODUCT:
        combined = boost_mask(combined)
    if MASK_CLOSE_SIZE >= 3:
        combined = combined.filter(ImageFilter.MaxFilter(MASK_CLOSE_SIZE))
        combined = combined.filter(ImageFilter.MinFilter(MASK_CLOSE_SIZE))
    if MASK_EXPAND_SIZE >= 3:
        combined = combined.filter(ImageFilter.MaxFilter(MASK_EXPAND_SIZE))
    combined = combined.point(lambda x: 0 if x <= ALPHA_ZERO_THRESHOLD else x)
    if EDGE_BLUR_RADIUS > 0:
        combined = combined.filter(ImageFilter.GaussianBlur(EDGE_BLUR_RADIUS))
    # Avoid an all-over faint haze from AI alpha residuals.
    combined = combined.point(lambda x: 0 if x < 8 else x)

    if SAVE_DEBUG_MASKS:
        combined.save(DEBUG_DIR / f"{stem}-final-mask.png")
    return combined


def prepare_logo():
    with Image.open(LOGO_PATH) as src:
        original = ImageOps.exif_transpose(src).convert("RGBA")

    # Scale to cover the whole canvas. A transparent logo will still have its
    # intrinsic transparent portions: those portions are not filled.
    if LOGO_COVER_CANVAS:
        ratio = max(CANVAS_SIZE[0] / original.width, CANVAS_SIZE[1] / original.height)
    else:
        ratio = min(CANVAS_SIZE[0] / original.width, CANVAS_SIZE[1] / original.height)
    size = (max(1, round(original.width * ratio)), max(1, round(original.height * ratio)))
    resized = original.resize(size, Image.Resampling.LANCZOS)
    watermark = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    watermark.paste(resized, ((600 - resized.width) // 2, (600 - resized.height) // 2))
    a = watermark.getchannel("A").point(lambda x: round(x * LOGO_OPACITY))
    watermark.putalpha(a)
    return watermark


_ai_upsampler = None
_ai_unavailable = False


def get_ai_upsampler():
    """Lazy-load Real-ESRGAN only when a low-resolution photo needs it."""
    global _ai_upsampler, _ai_unavailable
    if _ai_unavailable or not AI_SUPER_RESOLUTION:
        return None
    if _ai_upsampler is not None:
        return _ai_upsampler
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        if not AI_MODEL_FILE.is_file() and AI_AUTO_DOWNLOAD_MODEL:
            from urllib.request import urlretrieve
            AI_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            print("  Downloading Real-ESRGAN model (first use only)...")
            temp = AI_MODEL_FILE.with_suffix(".download")
            try:
                urlretrieve(AI_MODEL_URL, temp)
                temp.replace(AI_MODEL_FILE)
            finally:
                temp.unlink(missing_ok=True)
        if not AI_MODEL_FILE.is_file():
            raise FileNotFoundError(f"Missing AI weights: {AI_MODEL_FILE}")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        _ai_upsampler = RealESRGANer(
            scale=4, model_path=str(AI_MODEL_FILE), model=model,
            tile=256, tile_pad=16, pre_pad=0,
            half=AI_USE_HALF_PRECISION and torch.cuda.is_available(),
            gpu_id=0 if torch.cuda.is_available() else None,
        )
        print("  AI restoration enabled: Real-ESRGAN x4")
        return _ai_upsampler
    except Exception as exc:
        print(f"  WARNING: Real-ESRGAN could not start ({exc}). Using strong enhancement.")
        _ai_unavailable = True
        return None


def improve_product(crop, original_source_size):
    """Reconstruct weak product photos; preserve original transparency."""
    if not ENHANCE_SMALL_IMAGES or max(original_source_size) >= LOW_RESOLUTION_THRESHOLD:
        return crop

    alpha = crop.getchannel("A")
    rgb = crop.convert("RGB")
    # Avoid magnifying noise before enhancement.
    if STRONG_DENOISE:
        rgb = rgb.filter(ImageFilter.MedianFilter(3))

    upsampler = get_ai_upsampler()
    if upsampler is not None:
        try:
            import numpy as np
            # Real-ESRGAN expects BGR. Upscale color and alpha separately.
            # Keep memory bounded and do not reduce already-small sources.
            work = rgb.copy()
            if max(work.size) > AI_RECONSTRUCTION_MAX_SIDE:
                work.thumbnail((AI_RECONSTRUCTION_MAX_SIDE, AI_RECONSTRUCTION_MAX_SIDE),
                               Image.Resampling.LANCZOS)
            bgr = np.asarray(work, dtype=np.uint8)[:, :, ::-1].copy()
            upscaled, _ = upsampler.enhance(bgr, outscale=4)
            reconstructed = Image.fromarray(upscaled[:, :, ::-1].copy(), "RGB")
            new_alpha = alpha.resize(reconstructed.size, Image.Resampling.LANCZOS)
            reconstructed.putalpha(new_alpha)
            print("  Applied AI image reconstruction (4x)")
            return reconstructed
        except Exception as exc:
            warnings.warn(f"AI restoration failed; using high-strength fallback: {exc}")

    # High-strength deterministic fallback, without requiring AI dependencies.
    # Upscale before sharpening for better small-image display.
    if max(rgb.size) < 900:
        factor = min(4.0, 1250 / max(rgb.size))
        upsize = (max(1, round(rgb.width * factor)),
                  max(1, round(rgb.height * factor)))
        rgb = rgb.resize(upsize, Image.Resampling.LANCZOS)
    rgb = ImageEnhance.Contrast(rgb).enhance(CONTRAST_FACTOR)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.45,
                                             percent=UNSHARP_PERCENT, threshold=2))
    rgb = ImageEnhance.Sharpness(rgb).enhance(SHARPNESS_FACTOR)
    rgb.putalpha(alpha.resize(rgb.size, Image.Resampling.LANCZOS))
    print("  Applied strong image enhancement")
    return rgb


def render_image(source, sessions, watermark, filename):
    """Return an RGBA 600x600 image; no opaque background is introduced."""
    source = ImageOps.exif_transpose(source).convert("RGBA")
    original_size = source.size
    if max(source.size) > MAX_INPUT_DIMENSION:
        source.thumbnail((MAX_INPUT_DIMENSION, MAX_INPUT_DIMENSION), Image.Resampling.LANCZOS)

    # Existing PNG transparency is preserved instead of trying to replace it.
    original_alpha = source.getchannel("A")
    if original_alpha.getextrema()[0] < 255:
        mask = original_alpha
    else:
        mask = build_product_mask(source, sessions, filename)
    product = source.copy()
    product.putalpha(mask)

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("No product detected")
    crop = product.crop(bbox)
    crop = improve_product(crop, original_size)

    max_product_size = (600 - 2 * PRODUCT_PADDING, 600 - 2 * PRODUCT_PADDING)
    # Pillow's thumbnail will not enlarge; explicit resize allows small products
    # to occupy the intended display area.
    ratio = min(max_product_size[0] / crop.width, max_product_size[1] / crop.height)
    if ratio > 1 and not UPSCALE_SMALL_CROPS:
        ratio = 1
    target = (max(1, round(crop.width * ratio)), max(1, round(crop.height * ratio)))
    product_resized = crop.resize(target, Image.Resampling.LANCZOS)

    # Important: watermark FIRST, product SECOND.
    # Canvas begins fully transparent, and remains transparent outside both.
    canvas = watermark.copy()
    pos = ((600 - target[0]) // 2, (600 - target[1]) // 2)
    canvas.alpha_composite(product_resized, pos)
    return canvas


def process_all():
    if not PRODUCTS_DIR.is_dir():
        raise SystemExit(f"Missing products folder: {PRODUCTS_DIR}")
    if not LOGO_PATH.is_file():
        raise SystemExit(f"Missing watermark image: {LOGO_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_MASKS:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted((p for p in PRODUCTS_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
                   key=lambda p: p.name.casefold())
    if not files:
        print(f"No input images found in {PRODUCTS_DIR}")
        return

    logo = prepare_logo()
    sessions = load_models() if any(p.suffix.lower() != ".png" for p in files) else None
    # PNGs may also be opaque, so the models must be available if needed.
    count_ok = count_skip = count_errors = 0
    for index, file in enumerate(files, 1):
        destination = OUTPUT_DIR / f"{file.stem}.png"
        if SKIP_EXISTING and destination.exists():
            count_skip += 1
            continue
        print(f"[{index}/{len(files)}] {file.name}")
        try:
            with Image.open(file) as src:
                if sessions is None and ImageOps.exif_transpose(src).convert("RGBA").getchannel("A").getextrema()[0] == 255:
                    sessions = load_models()
                result = render_image(src, sessions, logo, file.stem)
            result.save(destination, format="PNG", optimize=True)
            print(f"  Saved: {destination.name} | {result.size} | {destination.stat().st_size / 1024:.0f} KB")
            result.close()
            count_ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            count_errors += 1
        gc.collect()
    print(f"Finished | success={count_ok}, skipped={count_skip}, errors={count_errors}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all()
