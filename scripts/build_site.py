"""Build the public demo site (`data/site/`) from finished run folders. UI only: reads exports, writes web assets.

    .venv\\Scripts\\python scripts/build_site.py                       # both demo runs
    .venv\\Scripts\\python scripts/build_site.py --run esri=data/interim/esri_s6

For every run it writes, under ``data/site/runs/<key>/``:

  data.json     the input check (29 measured checks), the QA KPIs regrouped **by pipeline process**
                rather than by stage, per-stage times with a 10-minute projection, coverage, zones,
                georeferencing, accuracy vs the reference surface, and the real export file sizes.
  assets/       one genuinely web-sized file **per output format**, derived from that format's real
                export so the page loads each through its own three.js loader:
                  mesh.glb / mesh.obj / mesh.fbx   the decimated mesh the pipeline already builds
                                                   for the viewer (vertex colour baked from the
                                                   texture: a UV atlas cannot survive decimation)
                  cloud.ply / cloud.las            the same subsample of the exported cloud, written
                                                   through plyfile and laspy, classification kept
                  dsm.png / ortho.png              rendered previews of the GeoTIFFs + their bounds

Nothing here touches the pipeline: it only reads what a run produced.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "data" / "site"
SITE_SRC = ROOT / "site"        # tracked hand-written page (index.html, app.js, styles.css), copied into SITE
BUILD = ROOT / "data" / "site_build"
DEFAULT_RUNS = {"esri": "data/interim/esri_s6", "dji": "data/interim/dji47_rep"}
CLOUD_POINTS = 400_000          # points kept for the web PLY/LAS (both the same subsample)
RASTER_MAX_PX = 1600            # longest side of a GeoTIFF preview

# What each pipeline process does, what its score covers, and what it does not measure. Written for a
# reader who will never open the source; the numbers beside them come from the run's own KPIs.
PROCESS_INFO: dict[str, dict[str, str]] = {
    "Frame selection": {
        "stage": "Ingest",
        "what": "Picks the frames to reconstruct from, targeting a set overlap between neighbours so every "
                "surface is seen from several angles without processing thousands of near-identical frames.",
        "covers": "Measured overlap between consecutive kept frames, how many frames were kept, and the "
                  "selection time against its budget.",
        "lacks": "It cannot know whether the flight itself covered the whole site: a strip the drone never "
                 "flew over is reported later, as a gap.",
    },
    "Blur gate": {
        "stage": "Ingest",
        "what": "Rejects frames whose motion blur would push wrong geometry into the reconstruction, and "
                "marks mildly soft frames for sharpening instead.",
        "covers": "The share of frames rejected, and whether the sharpness threshold adapted to the clip "
                  "rather than being fixed.",
        "lacks": "Blur is judged on the whole frame, so a sharp frame with one blurred moving object is "
                 "kept; the moving object is handled by dynamic masking instead.",
    },
    "Telemetry": {
        "stage": "Ingest",
        "what": "Reads the position log that came with the video, in whatever format it arrived, and puts it "
                "on the same clock as the frames.",
        "covers": "Which optional inputs were present (GPS, barometric altitude, attitude, focal length, RTK) "
                  "and how many fixes were aligned to frames.",
        "lacks": "It records what the telemetry claims; the accuracy of the GPS itself is measured later, "
                 "when the reconstructed camera path is compared against it.",
    },
    "Artifact suppression": {
        "stage": "Conditioning",
        "what": "Finds and softens compression blocking, the square edges a heavily compressed video leaves, "
                "which would otherwise be matched as if they were real corners.",
        "covers": "How many frames showed blocking, and the measured reduction on the frames that were "
                  "corrected, before and after.",
        "lacks": "It measures the blocking signal, not the downstream benefit; that is measured separately by "
                 "the degradation benchmark.",
    },
    "Illumination": {
        "stage": "Conditioning",
        "what": "Keeps brightness consistent across the flight, lifts shadow detail and marks shadowed pixels "
                "so they carry less weight when surfaces are matched.",
        "covers": "The exposure correction actually applied, whether any frame was driven to its limit, the "
                  "share of the frame in shadow and the share of blown-out pixels in the written frames.",
        "lacks": "Shadow detection has no per-pixel ground truth on real footage, so its precision is measured "
                 "on synthetic scenes rather than on this clip.",
    },
    "Dynamic masking": {
        "stage": "Conditioning",
        "what": "Masks out things that move, such as vehicles, so they neither anchor the camera solution nor "
                "leave smeared geometry in the model.",
        "covers": "The masked share of each frame and how many frames contained movers.",
        "lacks": "Without a semantic model installed it falls back to geometric consistency, which catches "
                 "movement but cannot name the object.",
    },
    "GPS conditioning": {
        "stage": "Conditioning",
        "what": "Removes impossible jumps from the position log and smooths the rest, so the track handed to "
                "georeferencing is physically plausible.",
        "covers": "Outliers removed, fixes smoothed, the altitude source chosen and the fastest speed observed.",
        "lacks": "Smoothing cannot add information the receiver never had; residual GPS error still sets the "
                 "absolute placement of the final model.",
    },
    "Sparse (SfM)": {
        "stage": "Reconstruction",
        "what": "Works out where the camera was for every frame by matching the same features across frames, "
                "then refines all positions together against the GPS track.",
        "covers": "The share of frames placed, how well the matched points reproject, how long each feature "
                  "track survives, and whether the solution broke into pieces.",
        "lacks": "A high score here means the solution is internally consistent; agreement with the real world "
                 "is the separate GPS check below.",
    },
    "Metric (vs telemetry)": {
        "stage": "Reconstruction",
        "what": "Checks the reconstruction against the world: camera positions against the GPS track, and the "
                "computed flying height against what the telemetry reported.",
        "covers": "Camera-centre agreement with GPS in metres, flying-height error as a percentage, the focal "
                  "length source, and a gate that caps the score if the solution disagrees with GPS.",
        "lacks": "GPS is the yardstick here, so this measures agreement with the telemetry, not absolute "
                 "truth; independent ground truth is the lidar comparison in Accuracy.",
    },
    "Track B (VGGT depth)": {
        "stage": "Reconstruction",
        "what": "Runs a learned depth model over the frames and pins its output to the measured geometry, "
                "adding detail in areas where stereo matching alone is thin.",
        "covers": "How many frames the depth was successfully anchored on, how tightly it agreed with the "
                  "measured points, and how many views confirm each fused point.",
        "lacks": "Learned depth is only accepted where it agrees with measurement; where it disagrees it is "
                 "refused, which protects accuracy but limits how much it can add.",
    },
    "Dense and mesh": {
        "stage": "Reconstruction",
        "what": "Turns the confirmed depth into a dense point cloud, then a continuous surface, then paints the "
                "photographs back onto it.",
        "covers": "Point and face counts, mesh soundness, whether texturing succeeded, and the share of the "
                  "texture that carries real image data.",
        "lacks": "Face count describes detail, not correctness; correctness is measured against the reference "
                 "surface in Accuracy.",
    },
    "Zones (§6.1–6.2)": {
        "stage": "Occlusion handling",
        "what": "Labels every part of the scene by how well it was actually observed: how many cameras saw it "
                "and from how wide an angle.",
        "covers": "The zone map itself, the share of the scene in each zone, and the view and angle statistics "
                  "behind the labels.",
        "lacks": "Zones describe observation quality, not the error in metres; the two are compared in Accuracy, "
                 "where the zones are shown to rank error correctly.",
    },
    "Zone 2 anchoring (§6.3)": {
        "stage": "Occlusion handling",
        "what": "Fills thinly observed areas using learned depth that is scaled and shifted to match the "
                "well-observed surface next to it, and never allowed to overwrite measured geometry.",
        "covers": "Frames accepted versus refused, the error on points deliberately held back from the fit, "
                  "and a re-check that no measured voxel was overwritten.",
        "lacks": "Filled surface is inferred, not measured; it is labelled as such in the model so a user can "
                 "exclude it.",
    },
    "Coverage and gaps (§6.4)": {
        "stage": "Occlusion handling",
        "what": "Reports what was reconstructed and, just as importantly, draws the outline of everything the "
                "cameras never saw instead of inventing it.",
        "covers": "The reconstructed share of visible ground, the number and area of gaps, and whether the gap "
                  "outlines were exported.",
        "lacks": "A gap is honest reporting, not a fix; closing one needs either another pass over that ground "
                 "or an assumption the system refuses to make.",
    },
    "Georeferencing": {
        "stage": "Georeferencing and export",
        "what": "Places the model on the map: fits the reconstruction to the GPS track, converts to a projected "
                "coordinate system and resolves what the altitudes were measured from.",
        "covers": "Horizontal and vertical agreement with GPS, the coordinate reference system written, the "
                  "altitude datum and, when ground control points are supplied, their independent check error.",
        "lacks": "Without ground control points the placement can be no better than the drone's own GPS.",
    },
    "Formats (re-opened)": {
        "stage": "Georeferencing and export",
        "what": "Writes all six required formats and then re-opens each one to confirm it is readable and "
                "carries the coordinates and layers it should.",
        "covers": "Every format verified by reading it back, the point cloud's classification, and the "
                  "confidence and zone layers travelling with the model.",
        "lacks": "Re-opening proves the file is valid and complete; how well third-party software displays it "
                 "is outside what can be measured here.",
    },
    "Coverage": {
        "stage": "Georeferencing and export",
        "what": "Carries the coverage figure measured during occlusion handling into the exported products, so "
                "the delivered files state how much of the scene they actually describe.",
        "covers": "The reconstructed percentage and the raster and vector products that record it.",
        "lacks": "Coverage counts ground the cameras saw; ground outside the flight path is a flight-planning "
                 "question, not a reconstruction one.",
    },
    "Runtime": {
        "stage": "Reconstruction",
        "what": "Tracks how long each step took against the time budget it was given, so an overrun degrades "
                "detail deliberately rather than running unbounded.",
        "covers": "Wall-clock time per step and the budget handed on to the next step.",
        "lacks": "Times are specific to the machine that ran them; the projection column re-states them for a "
                 "10-minute clip on this same hardware.",
    },
    "Time (§9)": {
        "stage": "Occlusion handling",
        "what": "Tracks how long occlusion handling took against its share of the overall time budget.",
        "covers": "Wall-clock time for zone classification and filling.",
        "lacks": "Times are specific to the machine that ran them.",
    },
}

# Order the processes in the order the video actually passes through them.
PROCESS_ORDER = [
    "Frame selection", "Blur gate", "Telemetry",
    "Artifact suppression", "Illumination", "Dynamic masking", "GPS conditioning",
    "Sparse (SfM)", "Metric (vs telemetry)", "Track B (VGGT depth)", "Dense and mesh", "Runtime",
    "Zones (§6.1–6.2)", "Zone 2 anchoring (§6.3)", "Coverage and gaps (§6.4)", "Time (§9)",
    "Georeferencing", "Formats (re-opened)", "Coverage",
]

FORMAT_INFO = {
    "obj": ("OBJ", "Wavefront mesh", "The textured surface as plain text geometry: the most widely accepted "
                                     "mesh format, opened by every 3D package."),
    "ply": ("PLY", "Polygon / point cloud", "Every reconstructed point with its colour and its per-point "
                                            "confidence and zone layers."),
    "las": ("LAS", "Lidar point cloud", "The survey-industry point format, classified into ground, above "
                                        "ground and low noise, and tiled for large sites."),
    "geotiff": ("GeoTIFF", "Georeferenced raster", "A height model and a true-colour orthophoto, both carrying "
                                                   "the coordinate system so they drop straight into any GIS."),
    "glb": ("GLB", "glTF binary", "The web and game-engine format: geometry and texture in one file, used by "
                                  "the viewer on this page."),
    "fbx": ("FBX", "Autodesk exchange", "The interchange format for Blender, 3ds Max, Maya and Unreal."),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a missing piece must not stop the build
        log(f"   !! {path.name}: {type(exc).__name__}: {exc}")
        return {}


def status_points(status: str) -> float | None:
    return {"pass": 1.0, "warn": 0.5, "fail": 0.0}.get(status)


def processes_from_scorecards(scorecards: dict) -> list[dict]:
    """Regroup every KPI of every stage by the process it belongs to (the KPI's own group)."""
    buckets: dict[str, dict] = {}
    for stage_key, card in scorecards.items():
        if not card or not card.get("kpis"):
            continue
        for kpi in card["kpis"]:
            name = kpi.get("group") or "Other"
            b = buckets.setdefault(name, {"process": name, "stage_key": stage_key,
                                          "stage": card.get("title", stage_key), "kpis": []})
            b["kpis"].append(kpi)
    out = []
    for name, b in buckets.items():
        scored = [k for k in b["kpis"] if status_points(k["status"]) is not None]
        b["score"] = round(100 * sum(status_points(k["status"]) for k in scored) / len(scored), 1) if scored else None
        b["counts"] = {s: sum(1 for k in b["kpis"] if k["status"] == s) for s in ("pass", "warn", "fail", "info")}
        info = PROCESS_INFO.get(name, {})
        b.update(what=info.get("what", ""), covers=info.get("covers", ""), lacks=info.get("lacks", ""))
        if info.get("stage"):
            b["stage"] = info["stage"]
        out.append(b)
    order = {n: i for i, n in enumerate(PROCESS_ORDER)}
    out.sort(key=lambda b: order.get(b["process"], 999))
    return out


def projection(report: dict) -> dict:
    """Restate this run's measured time for a 10-minute clip on the same machine."""
    time = report.get("time") or {}
    video_s = float(time.get("video_s") or 0) or None
    total_s = float(time.get("total_s") or 0)
    budget_s = float(time.get("budget_s") or 900)
    target_s = 600.0
    factor = (target_s / video_s) if video_s else None
    stages = time.get("stage_seconds") or {}
    return {
        "video_s": video_s, "total_s": total_s, "budget_s": budget_s, "target_video_s": target_s,
        "factor": factor,
        "projected_total_s": (total_s * factor) if factor else None,
        "stages": [{"stage": k, "seconds": v, "projected_s": (v * factor) if factor else None}
                   for k, v in stages.items()],
        "basis": "Times scale with the number of frames kept, which scales with the length of the flight, so "
                 "each measured time is multiplied by 10 minutes divided by this clip's length. Accuracy and "
                 "coverage figures are properties of the flight and do not change with clip length.",
    }


# ---------------------------------------------------------------------------- assets
def export_mesh_formats(run_dir: Path, out: Path) -> dict:
    """mesh.glb / mesh.obj / mesh.fbx from the decimated scene the pipeline already built."""
    import trimesh

    lite = run_dir / "qa" / "viewer" / "scene_lite.glb"
    if not lite.is_file():
        log("   !! no scene_lite.glb; skipping mesh formats")
        return {}
    shutil.copy2(lite, out / "mesh.glb")
    info = {"glb": {"file": "mesh.glb", "bytes": (out / "mesh.glb").stat().st_size}}

    mesh = trimesh.load(str(lite), force="mesh", process=False)
    v = np.asarray(mesh.vertices, np.float64)
    f = np.asarray(mesh.faces, np.int64)
    colours = getattr(mesh.visual, "vertex_colors", None)
    rgb = (np.asarray(colours)[:, :3] / 255.0) if colours is not None else np.full((len(v), 3), 0.75)
    centre = v.mean(axis=0)
    vc = v - centre                                     # local metres: keeps float precision in the browser

    obj = out / "mesh.obj"
    with obj.open("w", encoding="utf-8") as fh:
        fh.write("# decimated mesh with per-vertex colour baked from the texture atlas\n")
        for (x, y, z), (r, g, b) in zip(vc, rgb):
            fh.write(f"v {x:.4f} {y:.4f} {z:.4f} {r:.4f} {g:.4f} {b:.4f}\n")
        for a, b, c in f + 1:
            fh.write(f"f {a} {b} {c}\n")
    info["obj"] = {"file": "mesh.obj", "bytes": obj.stat().st_size}

    fbx = export_fbx(lite, out / "mesh.fbx")
    if fbx:
        info["fbx"] = {"file": "mesh.fbx", "bytes": fbx.stat().st_size}
    info["faces"], info["vertices"] = int(len(f)), int(len(v))
    info["centre"] = [round(float(c), 3) for c in centre]
    return info


def export_fbx(src_glb: Path, dest: Path) -> Path | None:
    """Convert through headless Blender, the same binary the export stage uses for its FBX."""
    blender = next(iter(sorted((ROOT / "tools").glob("blender*/blender.exe"))
                        + sorted((ROOT / "tools").glob("blender*/blender"))), None)
    if blender is None:
        log("   !! no Blender in tools/: skipping the FBX asset")
        return None
    script = dest.with_suffix(".convert.py")
    script.write_text(
        "import bpy, sys\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        f"bpy.ops.import_scene.gltf(filepath=r'{src_glb}')\n"
        f"bpy.ops.export_scene.fbx(filepath=r'{dest}', use_selection=False, colors_type='SRGB', "
        "path_mode='COPY', embed_textures=True, use_mesh_modifiers=False)\n", encoding="utf-8")
    proc = subprocess.run([str(blender), "-b", "--factory-startup", "--python", str(script)],
                          capture_output=True, text=True, timeout=900)
    script.unlink(missing_ok=True)
    if not dest.is_file():
        log(f"   !! Blender FBX failed: {(proc.stdout + proc.stderr)[-300:]}")
        return None
    return dest


def export_point_formats(run_dir: Path, out: Path, keep: int = CLOUD_POINTS) -> dict:
    """cloud.ply and cloud.las: the same even subsample of the exported cloud, written in each format."""
    import laspy

    src = run_dir / "export" / "cloud.las"
    if not src.is_file():
        log("   !! no cloud.las; skipping point formats")
        return {}
    with laspy.open(str(src)) as reader:
        header = reader.header
        total = header.point_count
        step = max(total // keep, 1)
        xs, ys, zs, cls, reds, greens, blues, zones, confs = ([] for _ in range(9))
        taken = 0
        for chunk in reader.chunk_iterator(1_000_000):
            sel = slice(None, None, step)
            xs.append(np.asarray(chunk.x)[sel]); ys.append(np.asarray(chunk.y)[sel]); zs.append(np.asarray(chunk.z)[sel])
            cls.append(np.asarray(chunk.classification)[sel])
            names = set(chunk.point_format.dimension_names)
            n = len(xs[-1])
            reds.append(np.asarray(chunk.red)[sel] if "red" in names else np.zeros(n, np.uint16))
            greens.append(np.asarray(chunk.green)[sel] if "green" in names else np.zeros(n, np.uint16))
            blues.append(np.asarray(chunk.blue)[sel] if "blue" in names else np.zeros(n, np.uint16))
            zones.append(np.asarray(chunk.zone)[sel] if "zone" in names else np.ones(n, np.uint8))
            confs.append(np.asarray(chunk.confidence)[sel] if "confidence" in names else np.zeros(n, np.float32))
            taken += n
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    classification = np.concatenate(cls).astype(np.uint8)
    r = (np.concatenate(reds) >> 8).astype(np.uint8)
    g = (np.concatenate(greens) >> 8).astype(np.uint8)
    b = (np.concatenate(blues) >> 8).astype(np.uint8)
    zone = np.concatenate(zones).astype(np.uint8)
    conf = np.concatenate(confs).astype(np.float32)
    centre = np.array([x.mean(), y.mean(), z.mean()])

    small = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    small.header.offsets, small.header.scales = header.offsets, header.scales
    small.x, small.y, small.z = x, y, z
    small.classification = classification
    small.red = np.concatenate(reds); small.green = np.concatenate(greens); small.blue = np.concatenate(blues)
    try:
        small.header.add_crs(header.parse_crs())
    except Exception:  # noqa: BLE001 - the preview stays usable without the embedded CRS
        pass
    small.write(str(out / "cloud.las"))

    from plyfile import PlyData, PlyElement

    local = np.c_[x - centre[0], y - centre[1], z - centre[2]].astype(np.float32)
    arr = np.empty(len(x), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = local[:, 0], local[:, 1], local[:, 2]
    arr["red"], arr["green"], arr["blue"] = r, g, b
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out / "cloud.ply"))

    classes, counts = np.unique(classification, return_counts=True)
    return {
        "ply": {"file": "cloud.ply", "bytes": (out / "cloud.ply").stat().st_size},
        "las": {"file": "cloud.las", "bytes": (out / "cloud.las").stat().st_size},
        "points_web": int(len(x)), "points_full": int(total),
        "centre": [round(float(c), 3) for c in centre],
        "classes": {int(c): int(n) for c, n in zip(classes, counts)},
        "zone_counts": {int(k): int(v) for k, v in zip(*np.unique(zone, return_counts=True))},
        "confidence_mean": round(float(conf.mean()), 3) if conf.size else None,
    }


def export_rasters(run_dir: Path, out: Path) -> dict:
    """dsm.png and ortho.png: honest renderings of the exported GeoTIFFs, plus their real georeferencing."""
    import rasterio
    from PIL import Image

    info: dict = {}
    for name, src_name in (("dsm", "dsm.tif"), ("ortho", "orthophoto.tif")):
        src = run_dir / "export" / src_name
        if not src.is_file():
            continue
        with rasterio.open(src) as ds:
            scale = min(1.0, RASTER_MAX_PX / max(ds.width, ds.height))
            w, h = max(int(ds.width * scale), 1), max(int(ds.height * scale), 1)
            bands = min(ds.count, 3)
            data = ds.read(list(range(1, bands + 1)), out_shape=(bands, h, w),
                           resampling=rasterio.enums.Resampling.average).astype(np.float32)
            nodata = ds.nodata
            bounds, crs = ds.bounds, (ds.crs.to_string() if ds.crs else None)
            res = (abs(ds.transform.a), abs(ds.transform.e))
        if bands >= 3:
            rgb = np.clip(data, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            alpha = np.where(rgb.sum(axis=2) > 0, 255, 0).astype(np.uint8)
            image = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
            stats = {}
        else:
            band = data[0]
            valid = np.isfinite(band) & (band != (nodata if nodata is not None else -9999))
            lo, hi = (np.percentile(band[valid], [2, 98]) if valid.any() else (0, 1))
            norm = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
            ramp = np.stack([np.clip(1.6 * norm - 0.3, 0, 1),          # terrain-style low->high ramp
                             np.clip(1.4 * norm ** 0.8, 0, 1) * 0.85 + 0.12,
                             np.clip(1.1 - 1.3 * norm, 0, 1) * 0.8 + 0.15], axis=-1)
            rgb = (ramp * 255).astype(np.uint8)
            image = Image.fromarray(np.dstack([rgb, np.where(valid, 255, 0).astype(np.uint8)]), "RGBA")
            stats = {"min_m": round(float(band[valid].min()), 2), "max_m": round(float(band[valid].max()), 2),
                     "ramp_low_m": round(float(lo), 2), "ramp_high_m": round(float(hi), 2)} if valid.any() else {}
        image.save(out / f"{name}.png", optimize=True)
        info[name] = {"file": f"{name}.png", "bytes": (out / f"{name}.png").stat().st_size,
                      "px": [w, h], "source_px": None, "crs": crs, "res_m": [round(r, 4) for r in res],
                      "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                      "source_file": src_name, "source_bytes": src.stat().st_size, **stats}
    return info


def build_run(key: str, run_dir: Path) -> dict:
    run_dir = (ROOT / run_dir) if not Path(run_dir).is_absolute() else Path(run_dir)
    log(f"== {key}: {run_dir}")
    out_dir = SITE / "runs" / key
    assets = out_dir / "assets"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets.mkdir(parents=True)

    report = read_json(run_dir / "qa" / "report.json")
    preflight = read_json(run_dir / "preflight" / "input_report.json")
    metadata = read_json(run_dir / "export" / "metadata.json")
    manifest = read_json(run_dir / "manifest.json")

    log("   mesh formats (glb / obj / fbx)")
    mesh = export_mesh_formats(run_dir, assets)
    log("   point formats (ply / las)")
    points = export_point_formats(run_dir, assets)
    log("   rasters (geotiff previews)")
    rasters = export_rasters(run_dir, assets)

    export_dir = run_dir / "export"
    files = {p.name: p.stat().st_size for p in sorted(export_dir.glob("*")) if p.is_file()}
    tiles = sorted((export_dir / "tiles").glob("*")) if (export_dir / "tiles").is_dir() else []

    data = {
        "key": key,
        "run": report.get("run") or run_dir.name,
        "video": Path(str(report.get("video") or preflight.get("video") or "")).name,
        "device": report.get("device"),
        "preset": report.get("preset"),
        "written": report.get("written"),
        "input_check": {
            "verdict": preflight.get("verdict"),
            "checks": preflight.get("checks") or [],
            "sync": preflight.get("sync") or {},
            "camera": preflight.get("camera") or {},
            "telemetry": (preflight.get("telemetry") or {}).get("summary") or {},
            "probe": preflight.get("probe") or {},
            "timing_s": preflight.get("timing_s") or {},
            "note": preflight.get("note"),
            "recommended": preflight.get("recommended") or {},
        },
        "processes": processes_from_scorecards(report.get("scorecards") or {}),
        "scorecards": {k: {"title": v.get("title"), "score": v.get("score"), "counts": v.get("counts")}
                       for k, v in (report.get("scorecards") or {}).items() if v},
        "time": report.get("time") or {},
        "projection": projection(report),
        "coverage": report.get("coverage") or {},
        "zones": report.get("zones") or {},
        "georeferencing": report.get("georeferencing") or {},
        "accuracy": report.get("accuracy_vs_reference") or {},
        "benchmarks": report.get("benchmarks") or {},
        "limitations": report.get("limitations") or [],
        "formats": {
            "produced": (report.get("formats") or {}).get("produced") or [],
            "info": FORMAT_INFO,
            "files": files,
            "tiles": len(tiles),
            "assets": {"mesh": mesh, "points": points, "rasters": rasters},
        },
        "metadata": {k: metadata.get(k) for k in ("crs", "coordinate_frames", "confidence", "coverage", "files")
                     if k in metadata},
        "manifest_stages": {k: {"status": v.get("status"), "duration_s": v.get("duration_s")}
                            for k, v in (manifest.get("stages") or {}).items()},
        "viewer": report.get("viewer") or {},
    }
    (out_dir / "data.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    log(f"   -> {out_dir.relative_to(ROOT)}  {total / 1e6:.1f} MB")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="KEY=PATH",
                    help="a run to publish (repeatable); default: " + ", ".join(f"{k}={v}" for k, v in DEFAULT_RUNS.items()))
    args = ap.parse_args()
    runs = dict(pair.split("=", 1) for pair in args.run) if args.run else DEFAULT_RUNS

    index = []
    for key, path in runs.items():
        try:
            data = build_run(key, Path(path))
        except Exception as exc:  # noqa: BLE001 - one bad run must not lose the others
            log(f"   !! {key} failed: {type(exc).__name__}: {exc}")
            continue
        index.append({"key": key, "run": data["run"], "video": data["video"],
                      "verdict": data["input_check"]["verdict"],
                      "coverage_pct": (data["coverage"] or {}).get("coverage_pct"),
                      "total_s": (data["time"] or {}).get("total_s"),
                      "video_s": (data["time"] or {}).get("video_s")})
    (SITE / "runs" / "index.json").write_text(json.dumps({"runs": index}, indent=1), encoding="utf-8")

    vendor_src, vendor_dst = BUILD / "vendor_add" / "addons", SITE / "vendor" / "addons"
    if vendor_src.is_dir():
        for src in vendor_src.rglob("*"):
            if src.is_file():
                dst = vendor_dst / src.relative_to(vendor_src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    base = ROOT / "viewer" / "vendor"
    for name in ("three.module.min.js", "LICENSE"):
        if (base / name).is_file():
            (SITE / "vendor").mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / name, SITE / "vendor" / name)
    for rel in ("addons/controls/OrbitControls.js", "addons/loaders/GLTFLoader.js", "addons/utils/BufferGeometryUtils.js"):
        src = base / rel
        if src.is_file():
            dst = SITE / "vendor" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    copied = 0
    for src in sorted(SITE_SRC.rglob("*")):
        if src.is_file():
            dst = SITE / src.relative_to(SITE_SRC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    log(f"   site source: {copied} file(s) copied from {SITE_SRC.relative_to(ROOT)}/")
    log(f"== done: {len(index)} run(s); site {sum(p.stat().st_size for p in SITE.rglob('*') if p.is_file()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
