import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import windows
import cv2
from pathlib import Path
from tqdm import tqdm

# ============================================================
# SVAMITVA V2 — Script 2: Tile Generator
# - Reprojects all villages to EPSG:32643
# - Cuts image + mask into 512x512 pairs
# - Stride=256 (50% overlap) — ~40K tiles
# - Keeps ALL tiles including background
# - Saves image as PNG, mask as PNG
# ============================================================

TIF_DIR    = Path(r"D:\villages")
MASK_DIR   = Path(r"D:\SVAMITVA_V2\masks")
OUT_IMG    = Path(r"D:\SVAMITVA_V2\tiles\images")
OUT_MASK   = Path(r"D:\SVAMITVA_V2\tiles\masks")
TEMP_DIR   = Path(r"D:\SVAMITVA_V2\tiles\temp")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS  = "EPSG:32643"
TILE_SIZE   = 512
STRIDE      = 410
NODATA_THR  = 0.7  # skip tile if >70% is nodata/black

def reproject_to_target(src_path, dst_path, target_crs):
    """Reproject TIF to target CRS if needed"""
    with rasterio.open(src_path) as src:
        if str(src.crs) == target_crs:
            return src_path  # already correct CRS

        print(f"  Reprojecting {src_path.name} → {target_crs}")
        transform, width, height = calculate_default_transform(
            src.crs, target_crs,
            src.width, src.height,
            *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs":       target_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
            "compress":  "lzw",
            "BIGTIFF":   "YES"
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source      = rasterio.band(src, band),
                    destination = rasterio.band(dst, band),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = transform,
                    dst_crs       = target_crs,
                    resampling    = Resampling.bilinear
                )
    return dst_path

def reproject_mask(mask_path, dst_path, ref_tif_path, target_crs):
    """Reproject mask to match reprojected TIF"""
    with rasterio.open(ref_tif_path) as ref:
        ref_transform = ref.transform
        ref_width     = ref.width
        ref_height    = ref.height
        ref_crs       = ref.crs

    with rasterio.open(mask_path) as src:
        if str(src.crs) == target_crs and \
           src.width == ref_width and \
           src.height == ref_height:
            return mask_path

        kwargs = src.meta.copy()
        kwargs.update({
            "crs":       target_crs,
            "transform": ref_transform,
            "width":     ref_width,
            "height":    ref_height,
            "compress":  "deflate"
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source        = rasterio.band(src, 1),
                destination   = rasterio.band(dst, 1),
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = ref_transform,
                dst_crs       = target_crs,
                resampling    = Resampling.nearest
            )
    return dst_path

def is_valid_tile(img_tile, mask_tile):
    """Check tile has enough valid pixels"""
    # Skip mostly black/nodata image tiles
    if np.mean(img_tile) < 8:
        return False
    # Check alpha channel if present
    valid_ratio = np.count_nonzero(img_tile[:,:,0]) / img_tile[:,:,0].size
    if valid_ratio < (1 - NODATA_THR):
        return False
    return True

print("=" * 70)
print("SVAMITVA V2 — Script 2: Tile Generator")
print(f"Tile: {TILE_SIZE}x{TILE_SIZE} | Stride: {STRIDE} | CRS: {TARGET_CRS}")
print("=" * 70)

tif_files = sorted(TIF_DIR.glob("*.tif"))
print(f"Found {len(tif_files)} villages\n")

total_saved   = 0
total_skipped = 0

for tif_path in tif_files:
    village_name = tif_path.stem.replace(" ", "_")

    # Find matching mask
    mask_path = MASK_DIR / f"{tif_path.stem}_mask.tif"
    if not mask_path.exists():
        print(f"\n[NO MASK] {tif_path.name} — skipping")
        continue

    print(f"\n{'='*55}")
    print(f"[TILE] {tif_path.name}")

    # Reproject TIF if needed
    reproj_tif  = TEMP_DIR / f"{village_name}_reproj.tif"
    reproj_mask = TEMP_DIR / f"{village_name}_mask_reproj.tif"

    try:
        final_tif  = reproject_to_target(
            tif_path, reproj_tif, TARGET_CRS
        )
        final_mask = reproject_mask(
            mask_path, reproj_mask,
            final_tif, TARGET_CRS
        )

        with rasterio.open(final_tif) as src:
            h, w   = src.height, src.width
            bands  = src.count
            has_alpha = bands >= 4
            print(f"  Image : {w} x {h} px | bands: {bands}")

        with rasterio.open(final_mask) as msrc:
            mh, mw = msrc.height, msrc.width
            print(f"  Mask  : {mw} x {mh} px")

        saved   = 0
        skipped = 0
        tile_id = 0

        rows = list(range(0, h - TILE_SIZE + 1, STRIDE))
        cols = list(range(0, w - TILE_SIZE + 1, STRIDE))
        total_possible = len(rows) * len(cols)

        pbar = tqdm(
            total=total_possible,
            desc=f"  Tiling"
        )

        with rasterio.open(final_tif) as src, \
             rasterio.open(final_mask) as msrc:

            for row in rows:
                for col in cols:
                    win = rasterio.windows.Window(
                        col, row, TILE_SIZE, TILE_SIZE
                    )

                    # Read image tile
                    img_data = src.read(
                        [1,2,3], window=win
                    ).transpose(1,2,0).astype(np.uint8)

                    # Skip nodata tiles
                    if not is_valid_tile(img_data, None):
                        skipped += 1
                        tile_id += 1
                        pbar.update(1)
                        continue

                    # Read mask tile
                    mask_data = msrc.read(
                        1, window=win
                    ).astype(np.uint8)

                    # Save filenames
                    fname = (f"{village_name}_"
                             f"{tile_id:06d}.png")

                    img_out  = OUT_IMG  / fname
                    mask_out = OUT_MASK / fname

                    # Save image as BGR for cv2
                    cv2.imwrite(
                        str(img_out),
                        cv2.cvtColor(img_data,
                                     cv2.COLOR_RGB2BGR)
                    )
                    cv2.imwrite(str(mask_out), mask_data)

                    saved   += 1
                    tile_id += 1
                    pbar.update(1)

        pbar.close()

        # Clean temp files
        if reproj_tif.exists() and reproj_tif != tif_path:
            reproj_tif.unlink()
        if reproj_mask.exists() and reproj_mask != mask_path:
            reproj_mask.unlink()

        total_saved   += saved
        total_skipped += skipped
        print(f"  Saved  : {saved:,} tiles")
        print(f"  Skipped: {skipped:,} nodata tiles")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print(f"Tiling complete!")
print(f"Total saved  : {total_saved:,} tiles")
print(f"Total skipped: {total_skipped:,} nodata tiles")
print(f"Images → {OUT_IMG}")
print(f"Masks  → {OUT_MASK}")
print("=" * 70)