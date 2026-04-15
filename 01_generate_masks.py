import os
import geopandas as gpd
import rasterio
from rasterio import features, windows
from shapely.geometry import box
import numpy as np
from pathlib import Path

# ============================================================
# SVAMITVA V2 — Script 1: 13-Class Mask Generator
# Optimized: spatial filtering + memory safe + GeoTIFF output
#
# CLASS MAP:
# 0=Background  1=RCC  2=Tiled  3=Tin  4=Other
# 5=Road  6=Road Centre  7=Water Body  8=Water Line
# 9=Water Point  10=Railway  11=Bridge  12=Utility
# ============================================================

TIF_DIR    = Path(r"D:\villages")
OUTPUT_DIR = Path(r"D:\SVAMITVA_V2\masks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CGDS = r"D:\chhattisgarh dataset\shp-file"
PBDS = r"D:\Punjab dataset\shp-file"

CG_KEYWORDS = [
    "badetumnar","murdanda","awapalli","nagul","madase",
    "ghotpal","kutru","samlur","bangapal","chhotetumar",
    "mofalnar","chintakonta"
]

CG_LAYERS = [
    ("Built_Up_Area_type.shp", "building"),
    ("Road.shp",               5),
    ("Road_Centre_Line.shp",   6),
    ("Water_Body.shp",         7),
    ("Water_Body_Line.shp",    8),
    ("Waterbody_Point.shp",    9),
    ("Railway.shp",           10),
    ("Bridge.shp",            11),
    ("Utility.shp",           12),
    ("Utility_Poly.shp",      12),
]

PB_LAYERS = [
    ("Built_Up_Area_typ.shp",  "building"),
    ("Road.shp",               5),
    ("Road_Centre_Line.shp",   6),
    ("Water_Body.shp",         7),
    ("Water_Body_Line.shp",    8),
    ("Waterbody_Point.shp",    9),
    ("Railway.shp",           10),
    ("Bridge.shp",            11),
    ("Utility.shp",           12),
    ("Utility_Poly_.shp",     12),
]

BLOCK_SIZE = 2048

def is_cg(village_stem):
    return any(k in village_stem.lower() for k in CG_KEYWORDS)
def load_shapes(village_stem, src_crs, village_bounds):
    cg      = is_cg(village_stem)
    shp_dir = CGDS if cg else PBDS
    layers  = CG_LAYERS if cg else PB_LAYERS
    dataset = "Chhattisgarh" if cg else "Punjab"
    print(f"  Dataset : {dataset}")

    # Convert village bounds to WGS84 for bbox filtering
    # (shapefiles may be in any CRS)
    from pyproj import Transformer
    try:
        transformer = Transformer.from_crs(
            src_crs, "EPSG:4326", always_xy=True
        )
        left, bottom = transformer.transform(
            village_bounds.left, village_bounds.bottom
        )
        right, top = transformer.transform(
            village_bounds.right, village_bounds.top
        )
        # Ensure correct order
        minx = min(left, right)
        miny = min(bottom, top)
        maxx = max(left, right)
        maxy = max(bottom, top)
        bbox_wgs84 = (minx, miny, maxx, maxy)
        print(f"  BBox WGS84: {minx:.3f},{miny:.3f} → "
              f"{maxx:.3f},{maxy:.3f}")
    except Exception as e:
        print(f"  Warning bbox transform: {e}")
        bbox_wgs84 = None

    shapes = []
    for shp_name, class_id in layers:
        shp_path = os.path.join(shp_dir, shp_name)
        if not os.path.exists(shp_path):
            continue
        try:
            # Read shapefile in its native CRS first
            if bbox_wgs84:
                # Get shapefile CRS to convert bbox correctly
                tmp = gpd.read_file(shp_path, rows=1)
                shp_crs = tmp.crs
                if shp_crs:
                    try:
                        t2 = Transformer.from_crs(
                            "EPSG:4326", shp_crs,
                            always_xy=True
                        )
                        bx1, by1 = t2.transform(
                            bbox_wgs84[0], bbox_wgs84[1]
                        )
                        bx2, by2 = t2.transform(
                            bbox_wgs84[2], bbox_wgs84[3]
                        )
                        bbox_shp = (
                            min(bx1,bx2), min(by1,by2),
                            max(bx1,bx2), max(by1,by2)
                        )
                        gdf = gpd.read_file(
                            shp_path, bbox=bbox_shp
                        ).to_crs(src_crs)
                    except Exception:
                        gdf = gpd.read_file(
                            shp_path
                        ).to_crs(src_crs)
                else:
                    gdf = gpd.read_file(
                        shp_path
                    ).to_crs(src_crs)
            else:
                gdf = gpd.read_file(
                    shp_path
                ).to_crs(src_crs)

            if gdf.empty:
                continue

            if class_id == "building":
                cnt = 0
                for rv, cid in [(1,1),(2,2),(3,3),(4,4)]:
                    sub = gdf[gdf["Roof_type"] == rv]
                    for geom in sub.geometry:
                        if geom is not None and geom.is_valid:
                            shapes.append((geom, cid))
                            cnt += 1
                print(f"  {shp_name}: {cnt} buildings")
            else:
                cnt = 0
                for geom in gdf.geometry:
                    if geom is not None and geom.is_valid:
                        shapes.append((geom, class_id))
                        cnt += 1
                if cnt > 0:
                    print(f"  {shp_name}: {cnt} features")

        except Exception as e:
            print(f"  Warning {shp_name}: {e}")

    print(f"  Total shapes: {len(shapes)}")
    return shapes

def generate_mask(tif_path, out_tif):
    """Generate mask GeoTIFF for one village"""
    tmp_tif = OUTPUT_DIR / f"{tif_path.stem}_tmp.tif"

    with rasterio.open(tif_path) as src:
        h, w      = src.height, src.width
        crs       = src.crs
        transform = src.transform
        bounds    = src.bounds
        print(f"  Size   : {w} x {h} px")
        print(f"  CRS    : {crs}")

    # Load only shapes within this village's extent
    shapes = load_shapes(tif_path.stem, crs, bounds)
    print(f"  Shapes : {len(shapes)} (spatially filtered)")

    if len(shapes) == 0:
        print("  WARNING: No shapes found for this village!")

    # Write mask block by block
    total = ((h // BLOCK_SIZE)+1) * ((w // BLOCK_SIZE)+1)
    done  = 0
    all_cls = set([0])
    nonzero = 0

    with rasterio.open(
        tmp_tif, "w",
        driver  = "GTiff",
        dtype   = "uint8",
        width   = w,
        height  = h,
        count   = 1,
        crs     = crs,
        transform = transform,
        compress  = "deflate"
    ) as dst:
        for row in range(0, h, BLOCK_SIZE):
            for col in range(0, w, BLOCK_SIZE):
                bh  = min(BLOCK_SIZE, h - row)
                bw  = min(BLOCK_SIZE, w - col)
                win = windows.Window(col, row, bw, bh)

                with rasterio.open(tif_path) as src:
                    wt = src.window_transform(win)

                blk = np.zeros((bh, bw), dtype=np.uint8)

                if shapes:
                    try:
                        features.rasterize(
                            shapes    = shapes,
                            out_shape = (bh, bw),
                            transform = wt,
                            out       = blk,
                            dtype     = np.uint8
                        )
                    except Exception as e:
                        pass  # empty block stays zero

                dst.write(blk, 1, window=win)

                # Track stats without loading full array
                u = np.unique(blk)
                all_cls.update(u.tolist())
                nonzero += int(np.sum(blk > 0))

                done += 1
                print(f"  Progress: {int(100*done/total)}%",
                      end="\r")
                del blk

    print()
    # Rename tmp to final
    if out_tif.exists():
        out_tif.unlink()
    tmp_tif.rename(out_tif)

    return sorted(all_cls), nonzero

# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("SVAMITVA V2 — Script 1: Mask Generator (Optimized)")
print("=" * 70)

tif_files = sorted(TIF_DIR.glob("*.tif"))
print(f"Found {len(tif_files)} TIF files\n")

completed = 0
failed    = []

for tif_path in tif_files:
    out_tif = OUTPUT_DIR / f"{tif_path.stem}_mask.tif"
    tmp_tif = OUTPUT_DIR / f"{tif_path.stem}_tmp.tif"

    if out_tif.exists():
        print(f"[SKIP] {tif_path.stem} — already done")
        completed += 1
        continue

    # Clean any incomplete temp file
    if tmp_tif.exists():
        tmp_tif.unlink()

    print(f"\n{'='*55}")
    print(f"[MASK] {tif_path.name}")

    try:
        all_cls, nonzero = generate_mask(tif_path, out_tif)
        print(f"  Classes : {all_cls}")
        print(f"  Non-zero: {nonzero:,} px")
        print(f"  Saved   : {out_tif.name}")
        completed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed.append(tif_path.name)
        if tmp_tif.exists():
            tmp_tif.unlink()

print("\n" + "=" * 70)
print(f"Completed : {completed}/{len(tif_files)}")
if failed:
    print(f"Failed    : {failed}")
print(f"Output    : {OUTPUT_DIR}")
print("=" * 70)