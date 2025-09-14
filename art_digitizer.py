#!/usr/bin/env python3
"""
Artwork Digitization & Portfolio Preparation
--------------------------------------------
A zero-style-transfer pipeline that cleans casual photos of artworks
(acrylic, color pencil, oil pastel, mixed media) into portfolio-ready images.

What it does (by design, non-destructive):
- Perspective correction (keeps shapes true to original)
- Subtle lighting normalization (reduce shadows/glare without repainting)
- Neutral background framing (white/gray/black) with consistent margins
- Mild color balancing (gray-world) and gentle sharpening (unsharp mask)
- Output: web JPEG @72 dpi + print PNG @300 dpi
- Simple metadata suggestion (title stub + tags from dominant colors)

Usage examples:
    python art_digitizer.py input.jpg --medium "Oil pastel" --bg white
    python art_digitizer.py folder/ --medium "Acrylic" --bg gray --margin 6
    python art_digitizer.py input.jpg --title "Moonlit Field" --year 2023 --dims "40x50 cm"

Notes:
- Requires: Python 3.9+, OpenCV (cv2), numpy, Pillow (PIL), scikit-image (optional but not required)
- Install: pip install opencv-python pillow numpy
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ---------------------------
# Utility functions
# ---------------------------

def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path: str, img_rgb: np.ndarray, dpi: Optional[Tuple[int, int]] = None, quality: int = 95):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(img_rgb)
    save_params = {}
    if dpi:
        save_params["dpi"] = dpi
    if p.suffix.lower() in [".jpg", ".jpeg"]:
        save_params.update(dict(quality=quality, optimize=True, progressive=True, subsampling=1))
    pil_img.save(str(p), **save_params)


def largest_quad_contour(binary: np.ndarray, min_area_ratio: float = 0.1) -> Optional[np.ndarray]:
    h, w = binary.shape[:2]
    img_area = float(h * w)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = abs(cv2.contourArea(approx))
            if area / img_area >= min_area_ratio:
                return approx.reshape(4, 2)
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    # order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = max(1, int(max(widthA, widthB)))
    maxH = max(1, int(max(heightA, heightB)))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def perspective_correct(image: np.ndarray) -> np.ndarray:
    # Downscale for speed
    h, w = image.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    quad = largest_quad_contour(edges, min_area_ratio=0.1)
    if quad is not None:
        quad = (quad / scale).astype(np.float32)
        warped = four_point_warp(image, quad)
        return warped
    # Fallback: gentle deskew using minimum area rectangle
    ys, xs = np.where(edges > 0)
    if xs.size:
        coords = np.column_stack((xs, ys)).astype(np.float32)
        angle = -cv2.minAreaRect(coords)[-1]
    else:
        angle = 0
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h2, w2) = image.shape[:2]
    center = (w2 // 2, h2 // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    return rotated


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    # Prevent oversaturation: normalize channel gains
    img_f = img.astype(np.float32) + 1e-6
    avg = img_f.mean(axis=(0,1))
    scale = avg.mean() / avg
    balanced = np.clip(img_f * scale, 0, 255).astype(np.uint8)
    return balanced


def normalize_lighting(img: np.ndarray) -> np.ndarray:
    # Work in LAB: CLAHE on L, mild
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def unsharp_mask(img: np.ndarray, radius: int = 2, amount: float = 1.0, threshold: int = 2) -> np.ndarray:
    pil = Image.fromarray(img)
    # Gentle sharpening; low threshold to preserve texture
    sharp = pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount*150), threshold=threshold))
    return np.array(sharp)


def add_neutral_frame(img: np.ndarray, bg: str = "white", margin_percent: int = 5) -> np.ndarray:
    assert bg in ("white", "black", "gray")
    h, w = img.shape[:2]
    margin = int(round(margin_percent / 100.0 * max(h, w)))
    new_h, new_w = h + 2*margin, w + 2*margin
    if bg == "white":
        canvas_color = (255, 255, 255)
    elif bg == "black":
        canvas_color = (0, 0, 0)
    else:
        canvas_color = (245, 245, 245)  # subtle gray
    canvas = np.full((new_h, new_w, 3), canvas_color, dtype=np.uint8)
    y0, x0 = margin, margin
    canvas[y0:y0+h, x0:x0+w] = img
    return canvas


def fit_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) == long_edge:
        return img
    if h >= w:
        new_h = long_edge
        new_w = int(w * (long_edge / h))
    else:
        new_w = long_edge
        new_h = int(h * (long_edge / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def suggest_title_and_tags(img: np.ndarray, medium: str) -> Tuple[str, List[str]]:
    # Very lightweight color-based tags (no style transfer)
    Z = img.reshape((-1,3)).astype(np.float32)
    K = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(int)
    color_names = []
    for c in centers:
        r,g,b = c
        # simple mapping
        if r>200 and g>200 and b>200: name="white"
        elif r<50 and g<50 and b<50: name="black"
        elif r>g and r>b: name="red" if r>150 else "warm"
        elif g>r and g>b: name="green"
        elif b>r and b>g: name="blue"
        elif r>180 and g>180 and b<100: name="yellow"
        else: name="mixed"
        color_names.append(name)
    # Deduplicate keeping order
    seen = set(); colors = [c for c in color_names if not (c in seen or seen.add(c))]
    title = f"Untitled ({medium})"
    tags = list(dict.fromkeys([medium.lower(), *colors, "artwork", "portfolio-ready", "texture", "true-to-life"]))
    return title, tags


def process_image(
    path: str,
    out_dir: str,
    medium: str,
    bg: str = "white",
    margin_percent: int = 5,
    web_long_edge: int = 2000,
    print_long_edge: int = 5000,
    title: Optional[str] = None,
    year: Optional[str] = None,
    dims: Optional[str] = None,
    series: Optional[str] = None
) -> dict:
    img = imread_rgb(path)

    # 1) Perspective correction
    img = perspective_correct(img)

    # 2) Subtle lighting normalization (no repainting)
    img = normalize_lighting(img)

    # 3) Gentle gray-world white balance
    img = gray_world_white_balance(img)

    # 4) Gentle sharpening (preserve stroke texture)
    img = unsharp_mask(img, radius=2, amount=0.8, threshold=2)

    # 5) Consistent framing on neutral background
    img_framed = add_neutral_frame(img, bg=bg, margin_percent=margin_percent)

    # 6) Output variants
    base = Path(path).stem
    safe_base = base.replace(" ", "_")

    # Metadata suggestion
    suggested_title, tags = suggest_title_and_tags(img, medium)
    final_title = title or suggested_title

    # Web
    web = fit_long_edge(img_framed, web_long_edge)
    web_name = f"{safe_base}_Portfolio_Web.jpg"
    web_path = str(Path(out_dir) / web_name)
    imwrite(web_path, web, dpi=(72,72), quality=88)

    # Print
    prn = fit_long_edge(img_framed, print_long_edge)
    prn_name = f"{safe_base}_Portfolio_Print.png"
    prn_path = str(Path(out_dir) / prn_name)
    imwrite(prn_path, prn, dpi=(300,300))

    # Metadata JSON
    meta = {
        "source_file": str(path),
        "outputs": {
            "web": web_path,
            "print": prn_path
        },
        "metadata": {
            "title": final_title,
            "medium": medium,
            "dimensions": dims or "",
            "year": year or "",
            "series": series or "",
            "description": f"A professionally digitized {medium.lower()} piece. Colors and textures preserved true-to-life; perspective corrected, lighting balanced, and framed on a neutral {bg} background.",
            "tags": tags
        },
        "settings": {
            "background": bg,
            "margin_percent": margin_percent,
            "web_long_edge": web_long_edge,
            "print_long_edge": print_long_edge,
            "sharpening": {"radius": 2, "amount": 0.8, "threshold": 2},
            "color_balance": "gray-world",
            "lighting": "LAB-CLAHE"
        }
    }
    meta_path = str(Path(out_dir) / f"{safe_base}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def list_images(p: Path) -> List[str]:
    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"}
    if p.is_file() and p.suffix.lower() in exts:
        return [str(p)]
    if p.is_dir():
        files = []
        for ext in exts:
            files.extend([str(x) for x in p.rglob(f"*{ext}")])
        return sorted(files)
    return []


def main():
    parser = argparse.ArgumentParser(description="Digitize artwork photos into portfolio-ready images.")
    parser.add_argument("input", help="Path to an image file or a folder of images.")
    parser.add_argument("--out", default="digitized", help="Output directory (default: digitized)")
    parser.add_argument("--medium", required=True, help="Medium, e.g., 'Oil pastel', 'Acrylic', 'Color pencil', 'Mixed media'")
    parser.add_argument("--bg", choices=["white","gray","black"], default="white", help="Neutral background color (default: white)")
    parser.add_argument("--margin", type=int, default=6, help="Margin as percent of long edge for consistent framing (default: 6)")
    parser.add_argument("--web_long_edge", type=int, default=2000, help="Target long edge (px) for web output (default: 2000)")
    parser.add_argument("--print_long_edge", type=int, default=5000, help="Target long edge (px) for print output (default: 5000)")
    parser.add_argument("--title", default=None, help="Optional final title")
    parser.add_argument("--year", default=None, help="Year (optional)")
    parser.add_argument("--dims", default=None, help="Dimensions, e.g., '30x40 cm' (optional)")
    parser.add_argument("--series", default=None, help="Series title (optional)")
    args = parser.parse_args()

    in_path = Path(args.input)
    items = list_images(in_path)
    if not items:
        raise SystemExit("No images found at the given input path.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for p in items:
        meta = process_image(
            path=p,
            out_dir=str(out_dir),
            medium=args.medium,
            bg=args.bg,
            margin_percent=args.margin,
            web_long_edge=args.web_long_edge,
            print_long_edge=args.print_long_edge,
            title=args.title,
            year=args.year,
            dims=args.dims,
            series=args.series
        )
        all_meta.append(meta)

    # Index manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(all_meta)} artwork(s).")
    print(f"Outputs saved under: {out_dir.resolve()}")
    print("For oil pastel works, remember you can pass --series 'Inner Beasts' if desired.")

if __name__ == "__main__":
    main()
