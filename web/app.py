from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import tempfile

import art_digitizer as ad


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Artwork Digitizer", version="1.0.0")
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "defaults": {
                "bg": "white",
                "margin": 6,
                "web_long_edge": 2000,
                "print_long_edge": 5000,
            },
        },
    )


@app.post("/process")
async def process(
    request: Request,
    files: List[UploadFile] = File(..., description="Artwork photo(s) to process"),
    medium: str = Form(..., description="e.g., Oil pastel, Acrylic"),
    bg: str = Form("white"),
    margin: int = Form(6),
    web_long_edge: int = Form(2000),
    print_long_edge: int = Form(5000),
    title: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    dims: Optional[str] = Form(None),
    series: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Create a persistent run directory to allow previews/downloads
    run_id = uuid.uuid4().hex[:8]
    run_dir = DATA_DIR / run_id
    upload_dir = run_dir / "uploads"
    out_dir = run_dir  # write outputs directly under run dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta: List[dict] = []

    # Save uploads and process each
    for i, up in enumerate(files):
        # Sanitize filename
        name = Path(up.filename or f"upload_{i}.jpg").name
        dest = upload_dir / name
        data = await up.read()
        if not data:
            continue
        dest.write_bytes(data)

        meta = ad.process_image(
            path=str(dest),
            out_dir=str(out_dir),
            medium=medium,
            bg=bg,
            margin_percent=int(margin),
            web_long_edge=int(web_long_edge),
            print_long_edge=int(print_long_edge),
            title=title,
            year=year,
            dims=dims,
            series=series,
        )
        all_meta.append(meta)

    if not all_meta:
        raise HTTPException(status_code=400, detail="No valid images to process")

    # Add a manifest at the root of the run directory
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Redirect to results page
    return RedirectResponse(url=f"/result/{run_id}", status_code=303)


@app.get("/result/{run_id}", response_class=HTMLResponse)
async def result(request: Request, run_id: str):
    run_dir = DATA_DIR / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read manifest: {e}")

    items = []
    for entry in manifest:
        try:
            web_abs = Path(entry["outputs"]["web"]).resolve()
            prn_abs = Path(entry["outputs"]["print"]).resolve()
            meta_abs = (run_dir / f"{Path(entry['source_file']).stem}_metadata.json").resolve()
            # Ensure paths are under run_dir
            web_rel = web_abs.relative_to(run_dir)
            prn_rel = prn_abs.relative_to(run_dir)
            meta_rel = meta_abs.relative_to(run_dir)
        except Exception:
            # Skip entries with unexpected paths
            continue
        items.append(
            {
                "title": entry.get("metadata", {}).get("title", "Untitled"),
                "medium": entry.get("metadata", {}).get("medium", ""),
                "web_url": f"/files/{run_id}/{web_rel.as_posix()}",
                "print_url": f"/files/{run_id}/{prn_rel.as_posix()}",
                "meta_url": f"/files/{run_id}/{meta_rel.as_posix()}",
                "filename": Path(entry.get("source_file", "")).name,
            }
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "run_id": run_id,
            "items": items,
        },
    )


@app.get("/download/{run_id}.zip")
async def download_zip(run_id: str):
    run_dir = DATA_DIR / run_id
    manifest_path = run_dir / "manifest.json"
    if not run_dir.exists() or not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(run_dir)))
    buf.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=digitized_{run_id}.zip"}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
