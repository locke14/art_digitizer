# Artwork Digitization & Portfolio Preparation

Zero–style‑transfer pipeline to turn casual photos of artworks (acrylic, color pencil, oil pastel, mixed media) into portfolio‑ready images. It preserves the work’s look while correcting perspective, normalizing lighting, gently balancing color, sharpening, and framing on a neutral background. Ships with both a CLI and a FastAPI web app.

## Features
- Perspective correction with quad detection and safe fallbacks
- Mild lighting normalization (LAB CLAHE) and gray‑world white balance
- Gentle sharpening to retain stroke texture
- Neutral background framing (white/gray/black) with consistent margins
- Two outputs per image: Web JPEG (72 dpi) and Print PNG (300 dpi)
- Lightweight metadata suggestions (title stub + color tags)
- Web UI for uploads, previews, and ZIP downloads

## Requirements
- Python 3.9+
- Packages:
  - Core: `opencv-python`, `pillow`, `numpy`
  - Web app: `fastapi`, `uvicorn[standard]`, `jinja2`, `python-multipart`

Install core + web deps:

```
pip install opencv-python pillow numpy
pip install -r web/requirements.txt
```

## CLI Usage
Run the CLI to process a single file or a whole folder. Outputs default to `digitized/`.

```
python art_digitizer.py input.jpg --medium "Oil pastel" --bg white
python art_digitizer.py folder/ --medium "Acrylic" --bg gray --margin 6
python art_digitizer.py input.jpg --title "Moonlit Field" --year 2023 --dims "40x50 cm"
```

Common flags:
- `--out`: output directory (default `digitized`)
- `--medium`: e.g., "Oil pastel", "Acrylic", "Color pencil", "Mixed media"
- `--bg`: one of `white`, `gray`, `black`
- `--margin`: percent of long edge for framing (default 6)
- `--web_long_edge`: web size long edge in px (default 2000)
- `--print_long_edge`: print size long edge in px (default 5000)
- Optional metadata: `--title`, `--year`, `--dims`, `--series`

Defaults by medium:
- Acrylic: softer white balance (0.65) and a small vibrancy lift (1.08)
- Oil/Pastel: gentler white balance (0.55) and vibrancy 1.06
- Others: white balance 0.8 and vibrancy 1.0

Outputs per source image:
- `*_Portfolio_Web.jpg` (72 dpi)
- `*_Portfolio_Print.png` (300 dpi)
- `*_metadata.json` (combined settings + suggestions)
- A batch `manifest.json` is written at the output root.

## Web App (FastAPI)
Launch a local web UI with uploads, previews, and per-file downloads.

```
uvicorn web.app:app --reload
```

Then open http://127.0.0.1:8000 and:
- Upload one or more images
- Choose `medium`, background, margin, and sizes
- Submit to process; you’ll be redirected to a results gallery
- Download individual files or the whole batch as a ZIP

Web app storage:
- Each run is stored under `web/data/<run_id>/` and served read‑only at `/files/<run_id>/...`.
- Clean up by deleting old folders under `web/data/` (ignored by Git).

## Deploy on Cloud Run (free tier)
Cloud Run runs a container and is a good fit for OpenCV. Storage is ephemeral, so results persist only while a given instance stays warm. For durable storage, integrate a bucket (GCS/S3) later.

1) Build container

```
gcloud builds submit --tag gcr.io/PROJECT_ID/art-digitizer
```

2) Deploy to Cloud Run

```
gcloud run deploy art-digitizer \
  --image gcr.io/PROJECT_ID/art-digitizer \
  --platform managed \
  --allow-unauthenticated \
  --region REGION \
  --memory 1Gi
```

Notes
- The app listens on `PORT` (default 8080). Cloud Run sets this env var.
- Runtime data dir is configurable via `DATA_DIR` (defaults to `/tmp/artdig_data` in the container). You can change it in the service config if needed.
- For persistent outputs, write to GCS and return links, or revert to “ZIP-only” responses.

### One-liner deploy scripts

- Bash (macOS/Linux/Git Bash):
  - `./scripts/deploy_cloud_run.sh PROJECT_ID REGION [SERVICE_NAME]`
  - Example: `./scripts/deploy_cloud_run.sh my-project us-central1 art-digitizer`

- PowerShell (Windows):
  - `powershell.exe -ExecutionPolicy Bypass -File scripts/deploy_cloud_run.ps1 -ProjectId PROJECT_ID -Region REGION -ServiceName art-digitizer`
  - Example: `powershell.exe -ExecutionPolicy Bypass -File scripts/deploy_cloud_run.ps1 -ProjectId my-project -Region us-central1 -ServiceName art-digitizer`

Both scripts:
- Enable required APIs
- Build the container via Cloud Build
- Deploy to Cloud Run
- Print the service URL at the end

## How It Works
- Entry point: `art_digitizer.py`
  - Core pipeline: `process_image(...)` applies perspective correction, lighting/white balance, sharpening, framing, and writes outputs + metadata.
- Web app: `web/app.py`
  - Routes: `/` (form), `/process` (runs a batch), `/result/{run_id}` (gallery), `/download/{run_id}.zip` (archive).

### Perspective robustness
The perspective step now rejects tiny or non‑convex quadrilaterals and clamps warp size to avoid degenerate outputs (e.g., “gray strip” artifacts). If an adequate quad isn’t found, it falls back to gentle deskew.

## Tips
- Large print sizes can take time; lower `--print_long_edge` for quick previews.
- For artworks shot very small in the frame, increase margin or move closer to fill more of the image; the detector prefers sizable quads.
- If deploying the web app publicly, add auth and a cleanup job for `web/data/`.

## Project Structure
- `art_digitizer.py` – CLI and image processing pipeline
- `web/app.py` – FastAPI application
- `web/templates/` – HTML templates (`index.html`, `result.html`)
- `web/data/` – Per‑run outputs (gitignored)
- `.gitignore` – excludes sample photos, outputs, and web data

## License
MIT License. See `LICENSE` for details.
