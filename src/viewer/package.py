"""Build the web viewer for one run's export (spec §8.4).

The viewer is a static folder that any browser opens through a local HTTP server
(``python -m src.cli view <run>``); no build step, three.js vendored in ``viewer/vendor``:

    index.html     the page (from ``viewer/index.html``), with scene.json inlined
    scene.glb      ONE mesh: positions (Y-up, local frame), the texture, and two custom
                   vertex attributes three.js exposes as ``_confidence`` and ``_zone``
    scene.json     stats panel, legend, offset/CRS for map coordinates, Zone 3 gap outlines
    vendor/        three.js r169 (MIT)

One mesh with per-vertex layers means the overlay toggle recolours the same geometry
instead of loading three models (the export's model.glb / model_confidence.glb /
model_zones.glb are 3 x 80 MB on Esri).

Everything is read from the export folder alone, so a copied export (box -> laptop) can
still be viewed: model.obj (textured, local frame), cloud.ply (confidence, zone, source per
point), model_zones.glb (zone per mesh vertex, including 3 = inferred), gaps.geojson,
metadata.json.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.core.logging import get_logger, log_downgrade, log_event
from src.export import writers

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "viewer"
SCENE_MARKER = "/*__SCENE_JSON__*/null"


# -- inputs ----------------------------------------------------------------------------------
def _read_cloud(path: Path) -> dict[str, np.ndarray] | None:
    """Local-frame points with their per-point layers from the export's cloud.ply."""
    if not path.is_file():
        return None
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    names = vertex.data.dtype.names
    out = {"xyz": np.c_[vertex["x"], vertex["y"], vertex["z"]].astype(np.float64)}
    for name in ("confidence", "zone", "source", "views"):
        if name in names:
            out[name] = np.asarray(vertex[name])
    return out


def _zones_from_glb(path: Path, vertices: np.ndarray) -> np.ndarray | None:
    """Zone (1/2/3) per viewer vertex from model_zones.glb's colours (nearest vertex)."""
    if not path.is_file():
        return None
    import trimesh
    from scipy.spatial import cKDTree

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    colours = np.asarray(loaded.visual.vertex_colors)[:, :3].astype(np.int32)
    # model_zones.glb was written Y-up; back to the local Z-up frame.
    zv = np.asarray(loaded.vertices) @ writers.Z_UP_TO_Y_UP[:3, :3]
    palette = np.array([writers.ZONE_COLOURS[z] for z in (1, 2, 3)], np.int32)
    zone_of = np.argmin(((colours[:, None, :] - palette[None]) ** 2).sum(-1), axis=1).astype(np.uint8) + 1
    _, idx = cKDTree(zv).query(vertices, k=1)
    return zone_of[idx]


def _nearest(cloud_xyz: np.ndarray, values: np.ndarray, vertices: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """(value of the nearest cloud point per vertex, True where that point is within max_dist)."""
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(cloud_xyz).query(vertices, k=1)
    return values[idx], dist <= max_dist


def _texture_jpeg(mesh, max_px: int, quality: int) -> tuple[bytes | None, tuple[int, int] | None]:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
    if image is None or getattr(visual, "uv", None) is None:
        return None, None
    from PIL import Image

    image = image.convert("RGB")
    if max(image.size) > max_px:
        scale = max_px / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), image.size


# -- glTF ------------------------------------------------------------------------------------
def write_scene_glb(path: Path, vertices_local: np.ndarray, faces: np.ndarray, uv: np.ndarray | None,
                    texture_jpeg: bytes | None, confidence: np.ndarray, zone: np.ndarray, extras: dict) -> Path:
    """One glTF 2.0 binary mesh: Y-up positions, optional texture, and custom attributes
    ``_CONFIDENCE`` (float 0-1, -1 = unknown) and ``_ZONE`` (float 0 unknown, 1/2/3)."""
    import pygltflib as g

    y_up = (np.c_[vertices_local, np.ones(len(vertices_local))] @ writers.Z_UP_TO_Y_UP.T)[:, :3].astype(np.float32)
    blobs: list[bytes] = []
    views: list[g.BufferView] = []
    accessors: list[g.Accessor] = []
    offset = 0

    def add(data: bytes, target: int | None = None) -> int:
        nonlocal offset
        pad = (-len(data)) % 4
        views.append(g.BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
        return len(views) - 1

    def accessor(array: np.ndarray, kind: str, component: int, target: int, bounds: bool = False) -> int:
        view = add(array.tobytes(), target)
        acc = g.Accessor(bufferView=view, componentType=component, count=len(array), type=kind)
        if bounds:
            acc.min, acc.max = array.min(0).tolist(), array.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    attributes = g.Attributes(POSITION=accessor(y_up, g.VEC3, g.FLOAT, g.ARRAY_BUFFER, bounds=True))
    if uv is not None and texture_jpeg is not None:
        # OBJ UV origin is bottom-left, glTF's top-left.
        gl_uv = np.c_[uv[:, 0], 1.0 - uv[:, 1]].astype(np.float32)
        attributes.TEXCOORD_0 = accessor(gl_uv, g.VEC2, g.FLOAT, g.ARRAY_BUFFER)
    custom = {"_CONFIDENCE": accessor(confidence.astype(np.float32), g.SCALAR, g.FLOAT, g.ARRAY_BUFFER),
              "_ZONE": accessor(zone.astype(np.float32), g.SCALAR, g.FLOAT, g.ARRAY_BUFFER)}
    for name, index in custom.items():
        setattr(attributes, name, index)
    indices = accessor(np.ascontiguousarray(faces, np.uint32).ravel(), g.SCALAR, g.UNSIGNED_INT,
                       g.ELEMENT_ARRAY_BUFFER)

    materials, textures, images, samplers = [], [], [], []
    if uv is not None and texture_jpeg is not None:
        images.append(g.Image(bufferView=add(texture_jpeg), mimeType="image/jpeg"))
        samplers.append(g.Sampler(magFilter=g.LINEAR, minFilter=g.LINEAR_MIPMAP_LINEAR))
        textures.append(g.Texture(sampler=0, source=0))
        materials.append(g.Material(pbrMetallicRoughness=g.PbrMetallicRoughness(
            baseColorTexture=g.TextureInfo(index=0), metallicFactor=0.0, roughnessFactor=1.0),
            doubleSided=True, extensions={"KHR_materials_unlit": {}}))
    else:
        materials.append(g.Material(pbrMetallicRoughness=g.PbrMetallicRoughness(
            baseColorFactor=[0.75, 0.75, 0.75, 1.0], metallicFactor=0.0, roughnessFactor=1.0), doubleSided=True))

    blob = b"".join(blobs)
    gltf = g.GLTF2(
        asset=g.Asset(version="2.0", generator="ps17 viewer package"),
        scene=0, scenes=[g.Scene(nodes=[0])], nodes=[g.Node(mesh=0, name="model")],
        meshes=[g.Mesh(primitives=[g.Primitive(attributes=attributes, indices=indices, material=0)])],
        accessors=accessors, bufferViews=views, buffers=[g.Buffer(byteLength=len(blob))],
        materials=materials, textures=textures, images=images, samplers=samplers,
        extensionsUsed=["KHR_materials_unlit"] if images else [], extras=extras,
    )
    gltf.set_binary_blob(blob)
    gltf.save_binary(str(path))
    return Path(path)


# -- gaps ------------------------------------------------------------------------------------
def gap_outlines(gaps_path: Path, meta: dict[str, Any], z_local: float, max_vertices: int,
                 height=None) -> list[dict[str, Any]]:
    """Zone 3 polygons from gaps.geojson in the local frame (x, y, z). ``height(xy)`` drapes them
    on the model (z per vertex, NaN where the model is absent); elsewhere they lie at ``z_local``."""
    if not gaps_path.is_file():
        return []
    doc = json.loads(gaps_path.read_text(encoding="utf-8"))
    frame = (meta.get("georeferencing") or {}).get("frame") or {}
    offset = np.asarray((meta.get("coordinate_frames") or {}).get("offset") or [0.0, 0.0, 0.0], float)
    lonlat = str((doc.get("properties") or {}).get("crs", "")).startswith("EPSG:4326")
    to_map = None
    if lonlat:
        if not frame.get("horizontal_epsg"):
            return []
        from pyproj import Transformer

        to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{frame['horizontal_epsg']}", always_xy=True)
    features = sorted(doc.get("features", []), key=lambda f: -float(f["properties"].get("area_m2", 0)))
    out, drawn = [], 0
    for feature in features:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        rings = []
        for polygon in polygons:
            for ring in polygon:
                xy = np.asarray(ring, float)
                if to_map is not None:
                    e, n = to_map.transform(xy[:, 0], xy[:, 1])
                    xy = np.c_[e, n] - offset[:2]
                elif not lonlat:
                    xy = xy - offset[:2]
                z = np.full(len(xy), z_local) if height is None else height(xy)
                z = np.where(np.isfinite(z), z, z_local)
                rings.append(np.round(np.c_[xy, z], 2).tolist())
        props = feature.get("properties", {})
        out.append({"area_m2": props.get("area_m2"), "kind": props.get("kind"), "rings": rings})
        drawn += sum(len(r) for r in rings)
        if drawn >= max_vertices:
            break
    return out


def _drape(vertices: np.ndarray, reach: float):
    """height(xy): the z of the nearest mesh vertex in plan (the mesh spans gaps as inferred faces)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(vertices[:, :2])

    def height(xy: np.ndarray) -> np.ndarray:
        dist, idx = tree.query(xy, k=1)
        return np.where(dist <= reach, vertices[idx, 2], np.nan)
    return height


# -- stats -----------------------------------------------------------------------------------
def scene_stats(meta: dict[str, Any], run_summary: dict[str, Any] | None) -> dict[str, Any]:
    """The stats panel (§8.4 item 5): time, coverage, accuracy, and what they mean."""
    g = meta.get("georeferencing") or {}
    frame = g.get("frame") or {}
    processing = meta.get("processing") or {}
    seconds = dict(processing.get("stage_seconds") or {})
    zones = meta.get("zones") or {}
    cov = meta.get("coverage") or {}
    gcp = g.get("gcp") or {}
    return {
        "video": Path(str(processing.get("video", ""))).name or None,
        "preset": processing.get("preset"),
        "stage_seconds": seconds,
        "total_seconds": round(sum(seconds.values()), 1) if seconds else None,
        "budget_seconds": ((run_summary or {}).get("budget") or {}).get("total_s"),
        "coverage_pct": cov.get("coverage_pct"),
        "zone_pct": {k: zones.get(f"zone{k}_pct") for k in (1, 2, 3)} if zones else None,
        "gaps": (zones.get("gaps") or {}) if zones else None,
        "holdout_error_median_m": zones.get("holdout_error_median_m") if zones else None,
        "cam_vs_gps_rms_m": g.get("rms_all_m"), "cam_vs_gps_rms_inliers_m": g.get("rms_inliers_m"),
        "horizontal_rms_m": g.get("horizontal_rms_m"), "vertical_rms_m": g.get("vertical_rms_m"),
        "gcp_check_rms_m": gcp.get("check_rms_m"),
        "crs": meta.get("crs"), "vertical_datum": frame.get("vertical_datum"),
        "formats": meta.get("formats_produced"), "formats_failed": sorted((meta.get("formats_failed") or {})),
        "accuracy_note": ("Positions: camera centres agree with GPS to the RMS above (telemetry-limited; "
                          "check points, when given, are the independent measure). Distances between two "
                          "Zone 1 points are more reliable than absolute position."),
    }


# -- the package -----------------------------------------------------------------------------
def build_viewer(export_dir: Path, out_dir: Path, vcfg: Any, *, run_summary: dict[str, Any] | None = None,
                 title: str | None = None) -> dict[str, Any]:
    """Write the viewer folder for ``export_dir``; returns {artifacts, metrics}."""
    export_dir, out_dir = Path(export_dir), Path(out_dir)
    meta_path = export_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no metadata.json in {export_dir}: run the export stage first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    mesh = None
    for name in ("model.obj", "model.glb"):
        if (export_dir / name).is_file():
            mesh = writers.load_mesh(export_dir / name)
            if name.endswith(".glb"):              # written Y-up; back to the local Z-up frame
                mesh.vertices = np.asarray(mesh.vertices) @ writers.Z_UP_TO_Y_UP[:3, :3]
            break
    cloud = _read_cloud(export_dir / "cloud.ply")
    if mesh is None:
        raise FileNotFoundError(f"no model.obj or model.glb in {export_dir}: the viewer needs a mesh")
    vertices, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)

    texture, tex_size = _texture_jpeg(mesh, int(vcfg.max_texture_px), int(vcfg.texture_jpeg_quality))
    uv = np.asarray(mesh.visual.uv) if texture is not None else None
    if texture is None:
        notes.append("untextured mesh: the viewer opens in the confidence view")

    max_dist = float(vcfg.layer_max_distance_m)
    confidence = np.full(len(vertices), -1.0, np.float32)
    zone = np.zeros(len(vertices), np.float32)
    if cloud is not None and "confidence" in cloud:
        values, near = _nearest(cloud["xyz"], cloud["confidence"].astype(np.float32), vertices, max_dist)
        confidence = np.where(near, values, -1.0).astype(np.float32)
    else:
        notes.append("no cloud.ply confidence: confidence view unavailable")
    zone_glb = _zones_from_glb(export_dir / "model_zones.glb", vertices)
    if zone_glb is not None:
        zone = zone_glb.astype(np.float32)
    elif cloud is not None and "zone" in cloud:
        values, near = _nearest(cloud["xyz"], cloud["zone"].astype(np.float32), vertices, max_dist)
        zone = np.where(near, values, 3.0).astype(np.float32)   # no measured point nearby = inferred
        notes.append("zones from cloud.ply (no model_zones.glb): vertices with no point within "
                     f"{max_dist:g} m shown as zone 3")
    else:
        notes.append("Stage 3 did not run for this export: zone view unavailable")

    offset = list((meta.get("coordinate_frames") or {}).get("offset") or [0.0, 0.0, 0.0])
    extras = {"offset": offset, "crs": meta.get("crs"), "frame": "local map frame, Y-up (east, up, -north)"}
    glb = write_scene_glb(out_dir / "scene.glb", vertices, faces, uv, texture, confidence, zone, extras)

    ground_z = (meta.get("coverage") or {}).get("ground_plane_z_m")
    z_local = (float(ground_z) - float(offset[2])) if ground_z is not None else float(np.percentile(vertices[:, 2], 5))
    gaps = gap_outlines(export_dir / "gaps.geojson", meta, z_local, int(vcfg.max_gap_vertices),
                        height=_drape(vertices, max_dist * 5))

    zone_known = zone > 0
    scene = {
        "title": title or (scene_stats(meta, run_summary).get("video") or export_dir.parent.name),
        "offset": offset, "crs": meta.get("crs"),
        "vertical_datum": (((meta.get("georeferencing") or {}).get("frame")) or {}).get("vertical_datum"),
        "georeferenced": bool(meta.get("crs")),
        "layers": {"texture": texture is not None, "confidence": bool((confidence >= 0).any()),
                   "zones": bool(zone_known.any())},
        "legend": {"confidence": meta.get("confidence"),
                   "zones": {"1": "well observed (measured)", "2": "thinly observed (measured or anchored fill)",
                             "3": "no measured support (inferred) - excluded from accuracy"},
                   "zone_colours": {str(k): list(v) for k, v in writers.ZONE_COLOURS.items()}},
        "gaps": gaps, "gap_plane_z": round(z_local, 2),
        "stats": scene_stats(meta, run_summary),
        "mesh": {"vertices": int(len(vertices)), "faces": int(len(faces)),
                 "texture_px": list(tex_size) if tex_size else None},
        "notes": notes,
    }
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=1, default=str), encoding="utf-8")
    page = (TEMPLATE / "index.html").read_text(encoding="utf-8")
    if SCENE_MARKER not in page:
        raise RuntimeError(f"{TEMPLATE / 'index.html'} lost its {SCENE_MARKER} marker")
    inline = json.dumps(scene, default=str).replace("</", "<\\/")
    (out_dir / "index.html").write_text(page.replace(SCENE_MARKER, inline), encoding="utf-8")
    if (out_dir / "vendor").exists():
        shutil.rmtree(out_dir / "vendor")
    shutil.copytree(TEMPLATE / "vendor", out_dir / "vendor")

    for note in notes:
        log_downgrade(log, "viewer layer", "the viewer without it", note)
    metrics = {"scene_mb": round(glb.stat().st_size / 1e6, 1), "faces": int(len(faces)),
               "vertices": int(len(vertices)), "layers": scene["layers"], "gaps_drawn": len(gaps),
               "confidence_known_pct": round(100.0 * float((confidence >= 0).mean()), 1),
               "zone_pct_vertices": {str(z): round(100.0 * float((zone == z).mean()), 1) for z in (1, 2, 3)},
               "notes": notes}
    log_event(log, logging.INFO, "viewer written", folder=str(out_dir), **{k: metrics[k] for k in ("scene_mb", "faces")})
    return {"artifacts": {"viewer": out_dir, "index": out_dir / "index.html", "scene": glb,
                          "scene_json": out_dir / "scene.json"}, "metrics": metrics}
