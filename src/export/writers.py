"""Writers for the six §8.2 formats.

Coordinates: meshes and PLY are written in the *local* map frame (UTM minus the offset in
``georef.json``; heights absolute), because float32 viewers jitter on 7-digit UTM northings.
LAS and GeoTIFF carry absolute map coordinates and a CRS in their headers.

Confidence is each point's number of confirming views (COLMAP/OpenMVS/hybrid visibility),
normalised by ``export.confidence_full_views``: it survives in PLY and LAS as a scalar and
in a second glTF file as vertex colours.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# -- inputs ---------------------------------------------------------------------
def read_dense(fused: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xyz, rgb uint8, views per point) from a COLMAP ``fused.ply`` (+ ``.vis``) or an OpenMVS cloud."""
    from plyfile import PlyData

    vertex = PlyData.read(str(fused))["vertex"]
    xyz = np.c_[vertex["x"], vertex["y"], vertex["z"]].astype(np.float64)
    names = vertex.data.dtype.names
    rgb = (np.c_[vertex["red"], vertex["green"], vertex["blue"]].astype(np.uint8) if "red" in names
           else np.full((len(xyz), 3), 180, np.uint8))
    if "view_indices" in names:
        views = np.fromiter((len(v) for v in vertex["view_indices"]), dtype=np.int32, count=len(xyz))
    elif Path(str(fused) + ".vis").exists():
        views = read_vis_counts(Path(str(fused) + ".vis"), len(xyz))
    else:
        views = np.ones(len(xyz), np.int32)
    return xyz, rgb, views


def read_vis_counts(path: Path, expected: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32, offset=8)
    n = struct.unpack("<Q", Path(path).open("rb").read(8))[0]
    counts = np.empty(n, np.int32)
    pos = 0
    for i in range(n):  # sequential by construction: each count says how far to skip
        counts[i] = raw[pos]
        pos += 1 + raw[pos]
    return counts if n == expected else np.ones(expected, np.int32)


def confidence_from_views(views: np.ndarray, full_views: int) -> np.ndarray:
    return np.clip(views / max(full_views, 1), 0.0, 1.0).astype(np.float32)


# -- point clouds -------------------------------------------------------------------
# Stage 3 per-point layers (uint8): zone 1/2 (well / thinly observed) and source 0/1 (MVS / anchored
# monocular fill). Written when the run has them; ``extra`` maps name -> (array, description).
POINT_LAYERS = {"zone": "Stage 3 zone: 1 well observed, 2 thinly observed",
                "source": "0 multi-view stereo, 1 monocular depth anchored to zone 1"}


def write_ply(path: Path, local_xyz: np.ndarray, rgb: np.ndarray, views: np.ndarray, confidence: np.ndarray,
              extra: dict[str, np.ndarray] | None = None) -> Path:
    from plyfile import PlyData, PlyElement

    extra = extra or {}
    vertex = np.empty(len(local_xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
                                             ("blue", "u1"), ("views", "u1"), ("confidence", "f4")]
                      + [(name, "u1") for name in extra])
    for name, values in extra.items():
        vertex[name] = values
    for c, k in enumerate("xyz"):
        vertex[k] = local_xyz[:, c]
    for c, k in enumerate(("red", "green", "blue")):
        vertex[k] = rgb[:, c]
    vertex["views"] = np.clip(views, 0, 255)
    vertex["confidence"] = confidence
    PlyData([PlyElement.describe(vertex, "vertex")], byte_order="<").write(str(path))
    return Path(path)


def write_las(path: Path, map_xyz: np.ndarray, rgb: np.ndarray, views: np.ndarray, confidence: np.ndarray,
              crs_string: str | None, scale: list[float], point_format: int = 3,
              extra: dict[str, np.ndarray] | None = None) -> Path:
    import laspy
    from pyproj import CRS

    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.asarray(scale, dtype=np.float64)
    header.offsets = np.floor(map_xyz.min(0))
    header.add_extra_dim(laspy.ExtraBytesParams(name="views", type=np.uint8, description="confirming views"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32, description="views / full"))
    for name in extra or {}:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8,
                                                    description=POINT_LAYERS.get(name, name)[:31]))
    if crs_string:
        # LAS 1.4 carries the CRS as OGC WKT, which (unlike GeoTIFF keys) can express the
        # compound UTM + EGM96 height CRS; laspy's add_crs() only handles EPSG-coded CRSs.
        header.vlrs.append(laspy.vlrs.known.WktCoordinateSystemVlr(CRS.from_user_input(crs_string).to_wkt()))
        header.global_encoding.wkt = True
    las = laspy.LasData(header)
    las.x, las.y, las.z = map_xyz[:, 0], map_xyz[:, 1], map_xyz[:, 2]
    las.red, las.green, las.blue = (rgb[:, c].astype(np.uint16) * 257 for c in range(3))
    las.views = np.clip(views, 0, 255).astype(np.uint8)
    las.confidence = confidence
    for name, values in (extra or {}).items():
        setattr(las, name, np.asarray(values, np.uint8))
    las.classification = np.ones(len(map_xyz), np.uint8)  # 1 = unclassified
    las.write(str(path))
    return Path(path)


def write_las_tiles(folder: Path, map_xyz, rgb, views, confidence, crs_string, scale, tile_m: float,
                    extra: dict[str, np.ndarray] | None = None) -> list[Path]:
    """One LAS per ``tile_m`` x ``tile_m`` map cell (§8.2 tiling, the Scalability criterion)."""
    folder.mkdir(parents=True, exist_ok=True)
    keys = np.floor(map_xyz[:, :2] / tile_m).astype(np.int64)
    out = []
    for key in np.unique(keys, axis=0):
        sel = np.all(keys == key, axis=1)
        out.append(write_las(folder / f"tile_{int(key[0] * tile_m)}_{int(key[1] * tile_m)}.las",
                             map_xyz[sel], rgb[sel], views[sel], confidence[sel], crs_string, scale,
                             extra={k: v[sel] for k, v in (extra or {}).items()}))
    return out


# -- meshes ---------------------------------------------------------------------------
def mesh_is_broken(vertices: np.ndarray, cloud: np.ndarray | None, factor: float) -> str | None:
    """Why a mesh must not be exported, or None. Non-finite vertices, or vertices farther than
    ``factor`` x the dense cloud's extent outside the cloud's bounding box (a degenerate Stage 4
    mesh: float32 overflow in glTF, a Blender hang in FBX; T-1)."""
    if not len(vertices):
        return "the mesh has no vertices"
    bad = ~np.isfinite(vertices).all(axis=1)
    if bad.any():
        return f"{int(bad.sum()):,} of {len(vertices):,} vertices are not finite"
    if cloud is None or not len(cloud) or factor <= 0:
        return None
    lo, hi = cloud.min(0), cloud.max(0)
    reach = factor * max(float(np.linalg.norm(hi - lo)), 1.0)
    far = ((vertices < lo - reach) | (vertices > hi + reach)).any(axis=1)
    if far.any():
        worst = float(np.abs(vertices[far]).max())
        return (f"{int(far.sum()):,} of {len(vertices):,} vertices lie more than {factor:g}x the cloud's "
                f"extent ({reach:,.0f} m) outside it (largest coordinate {worst:.3g})")
    return None


def load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    return loaded


def write_obj(path: Path, mesh) -> Path:
    """Untextured mesh as OBJ (``trimesh``), already in the local map frame."""
    mesh.export(str(path))
    return Path(path)


def transform_obj(src: Path, dst: Path, georef) -> Path:
    """Textured OBJ into the local map frame, losslessly: only ``v``/``vn`` lines change;
    the MTL and texture images are copied byte for byte (a trimesh round trip re-encoded
    OpenMVS's 2.7 MB texture into a 0.9 MB JPEG)."""
    src, dst = Path(src), Path(dst)
    rot = georef.scale * georef.rotation
    trans = georef.translation
    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("v "):
                x, y, z = (float(t) for t in line.split()[1:4])
                p = rot @ (x, y, z) + trans
                fout.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            elif line.startswith("vn "):
                n = georef.rotation @ tuple(float(t) for t in line.split()[1:4])
                fout.write(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n")
            elif line.startswith("mtllib "):
                name = line.split(maxsplit=1)[1].strip()
                shutil.copy2(src.parent / name, dst.parent / name)
                for mline in (src.parent / name).read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = mline.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].lower().startswith("map_") and (src.parent / parts[1]).exists():
                        shutil.copy2(src.parent / parts[1], dst.parent / parts[1])
                fout.write(line)
            else:
                fout.write(line)
    return dst


# glTF is Y-up, right-handed; map frames are Z-up (east, north, up).
Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def write_glb(path: Path, mesh, extras: dict[str, Any]) -> Path:
    import pygltflib

    y_up = mesh.copy()
    y_up.apply_transform(Z_UP_TO_Y_UP)
    y_up.export(str(path))
    gltf = pygltflib.GLTF2().load(str(path))
    gltf.extras = extras  # root extras: pygltflib drops asset.extras on save
    gltf.save(str(path))
    return Path(path)


def confidence_colours(mesh, cloud_local: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Per-vertex RGBA from the nearest dense point's confidence: red (1 view) -> green (full)."""
    from scipy.spatial import cKDTree

    _, idx = cKDTree(cloud_local).query(np.asarray(mesh.vertices), k=1)
    c = confidence[idx]
    return np.c_[(255 * (1 - c)), (255 * c), np.full_like(c, 40), np.full_like(c, 255)].astype(np.uint8)


def write_confidence_glb(path: Path, mesh, cloud_local: np.ndarray, confidence: np.ndarray, extras: dict) -> Path:
    import trimesh

    coloured = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    coloured.visual = trimesh.visual.ColorVisuals(coloured, vertex_colors=confidence_colours(mesh, cloud_local,
                                                                                             confidence))
    return write_glb(path, coloured, {**extras, "layer": "confidence (views / full), red low -> green high"})


ZONE_COLOURS = {1: (31, 136, 61), 2: (191, 135, 0), 3: (207, 34, 46)}


def write_zones_glb(path: Path, mesh, face_zone: np.ndarray, extras: dict) -> Path:
    """Stage 3 layer: vertices coloured by the worst zone of their faces (3 = inferred, no support)."""
    import trimesh

    faces = np.asarray(mesh.faces)
    vertex_zone = np.ones(len(mesh.vertices), np.uint8)
    for corner in range(3):
        np.maximum.at(vertex_zone, faces[:, corner], face_zone)
    colours = np.array([ZONE_COLOURS.get(int(z), (128, 128, 128)) + (255,) for z in range(4)], np.uint8)[vertex_zone]
    layer = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=faces, process=False)
    layer.visual = trimesh.visual.ColorVisuals(layer, vertex_colors=colours)
    legend = {"1": "well observed (zone 1)", "2": "thinly observed (zone 2)",
              "3": "no measured support: inferred surface (zone 3) — excluded from accuracy"}
    return write_glb(path, layer, {**extras, "layer": "Stage 3 zones", "legend": legend,
                                   "colours": {str(k): list(v) for k, v in ZONE_COLOURS.items()}})


# -- FBX via headless Blender (§8.3: the fragile one) ----------------------------------------
BLENDER_SCRIPT = """
import bpy, sys
src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
if hasattr(bpy.ops.wm, "obj_import"):
    bpy.ops.wm.obj_import(filepath=src, forward_axis="Y", up_axis="Z")
else:
    bpy.ops.import_scene.obj(filepath=src, axis_forward="Y", axis_up="Z")
bpy.ops.export_scene.fbx(filepath=dst, path_mode="COPY", embed_textures=True, axis_forward="-Z", axis_up="Y")
"""


def find_blender(configured: str | None) -> Path | None:
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates += sorted((ROOT / "tools").glob("blender*/blender")) + sorted((ROOT / "tools").glob("blender*/blender.exe"))
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    return next((c for c in candidates if c.exists()), None)


def write_fbx(path: Path, obj_path: Path, blender: Path, timeout_s: int = 600) -> Path:
    script = Path(path).with_suffix(".blender.py")
    script.write_text(BLENDER_SCRIPT, encoding="utf-8")
    result = subprocess.run([str(blender), "-b", "--factory-startup", "--python", str(script), "--",
                             str(obj_path), str(path)], capture_output=True, text=True, timeout=timeout_s)
    script.unlink(missing_ok=True)
    if result.returncode != 0 or not Path(path).exists():
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"Blender exited {result.returncode}: {' | '.join(tail)}")
    return Path(path)


def file_entry(path: Path, kind: str, **extra) -> dict[str, Any]:
    path = Path(path)
    return {"format": kind, "path": str(path), "bytes": path.stat().st_size if path.exists() else 0, **extra}


def dump_json(path: Path, data: dict) -> Path:
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return Path(path)
