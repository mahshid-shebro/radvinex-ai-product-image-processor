# Radvinex AI Product Image Processor

Batch-process product photography using three background-removal AI models, add a brand logo, and export standardized WebP images for e-commerce.

## Features

- Background removal using **BiRefNet General**, **BiRefNet Massive**, and **ISNet General Use** via `rembg`.
- Mask combination and recovery steps designed to preserve fine product details.
- Batch processing of JPG, JPEG, PNG, WEBP, BMP, TIFF, and TIF inputs.
- Consistent **600 × 600 px** output canvas, with a product region and bottom-left logo.
- WebP compression that targets **95 KiB or less**; if necessary the script reduces effective detail while preserving output dimensions. Files that cannot meet the limit raise an error.
- Optional saved diagnostic masks.

## Setup

**Python 3.11** is a practical starting point. The script is designed for local use and may need substantial memory, disk space, and time when loading its three models.

```bash
python -m venv .venv
```

Activate the environment:

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

> Note: The exact dependency versions and model-loading behavior have not been runtime-tested for this release. `rembg` downloads AI model weights when needed, so the initial run requires network access. Some environments may require additional model-specific dependencies or a compatible inference provider.

## Usage

1. Place source product images in `products/`.
2. Keep your logo at `logo.png`, or replace it with your own PNG logo.
3. Run:

```bash
python main.py
```

4. Find the resulting `.webp` images in `output/`.

### Project layout

```text
radvinex-ai-product-image-processor/
├── main.py
├── logo.png
├── requirements.txt
├── README.md
├── .gitignore
├── products/          # input images (not committed)
└── output/            # generated images (not committed)
```

## Customization

Edit constants near the top of `main.py` to adjust the image dimensions, product position, logo position, model names, compression target, or debugging options. `SAVE_DEBUG_MASKS = True` writes masks to `debug-masks/`.

## Important notes

- The code is for local batch processing; it is **not** a Photoshop plugin.
- A white RGB canvas is used for the final WebP image; generated images are **not transparent**.
- Runtime performance and output quality vary with the product images and AI model availability.
- A logo is included to reproduce the supplied setup; replace it when using the project for another brand. Permission to redistribute the logo and the preferred software license should be confirmed before public publication.
- No open-source license has been assigned yet. Choose a license (for example, MIT) before inviting reuse or modification.

## فارسی

این پروژه برای آماده‌سازی گروهی تصاویر محصولات فروشگاهی است: حذف پس‌زمینه با سه مدل هوش مصنوعی، حفظ لبه‌های محصول، قرار دادن لوگو و خروجی WebP در ابعاد ۶۰۰ در ۶۰۰ پیکسل. تصاویر ورودی را در پوشه `products` قرار دهید و با `python main.py` اجرا کنید. خروجی در پوشه `output` ذخیره می‌شود.
