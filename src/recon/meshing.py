"""Point-cloud and mesh utilities for the reconstruction tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ply_counts(path: Path) -> dict[str, int]:
    """Vertex and face counts from a PLY header, without loading the mesh."""
    counts: dict[str, int] = {}
    with Path(path).open("rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element "):
                _, name, n = line.split()
                counts[{"vertex": "vertices", "face": "faces"}.get(name, name)] = int(n)
            if line == "end_header":
                break
    return counts


def read_ply_xyz(path: Path) -> np.ndarray:
    """Vertex positions of a PLY, including OpenMVS clouds with per-point view lists
    (which ``trimesh`` rejects as "unexpected length")."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    return np.c_[vertex["x"], vertex["y"], vertex["z"]].astype(np.float64)


def clean_mesh(src: Path, dst: Path, min_component_fraction: float) -> dict[str, int]:
    """Drop non-finite vertices, degenerate faces and small fragments.

    Used on the Poisson fallback path only. On the box, pycolmap-cuda12's Poisson gave
    1.75 M vertices for 1.82 M faces with NaN coordinates (S4-6); ``trimesh.split``
    stalled on it, so fragments are labelled with one sparse connected-components pass.
    """
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mesh = trimesh.load(src, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {src.name} has no faces")
    stats = {"vertices_in": len(mesh.vertices), "faces_in": len(mesh.faces)}
    finite = np.isfinite(mesh.vertices).all(axis=1)
    stats["nonfinite_vertices"] = int((~finite).sum())
    faces = np.asarray(mesh.faces)
    keep = finite[faces].all(axis=1)
    keep &= (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]

    n = len(mesh.vertices)
    rows = np.concatenate([faces[:, 0], faces[:, 1]])
    cols = np.concatenate([faces[:, 1], faces[:, 2]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    face_label = labels[faces[:, 0]]
    counts = np.bincount(face_label, minlength=n_comp)
    big = counts >= min_component_fraction * max(counts.max(), 1)
    stats.update(components_in=int((counts > 0).sum()), components_kept=int(big.sum()))

    mask = keep.copy()
    mask[keep] = big[face_label]
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    mesh.export(dst)
    stats.update(vertices=len(mesh.vertices), faces=len(mesh.faces))
    return stats


def texture_quality(obj_path: Path, empty_color: tuple[int, int, int], tolerance: int = 24,
                    black_level: int = 8) -> dict:
    """Share of the used texture atlas that is black (S4-11), from the OBJ's MTL ``map_Kd`` images.

    OpenMVS fills unused atlas space with ``empty_color``; everything else is face texture or
    patch padding copied from the photos, so black there means the faces lost their colour."""
    import cv2

    obj_path = Path(obj_path)
    mtls = []
    with obj_path.open("r", encoding="utf-8", errors="replace") as fh:
        for _, line in zip(range(200), fh):                   # mtllib sits in the header
            if line.startswith("mtllib"):
                mtls.append(obj_path.parent / line.split(None, 1)[1].strip())
    images = []
    for mtl in (m for m in mtls if m.is_file()):
        for line in mtl.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().lower().startswith("map_kd"):
                images.append(obj_path.parent / line.split(None, 1)[1].strip())
    used = black = 0
    for image in images:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = img[:, :, ::-1].astype(np.int16)
        in_use = np.abs(rgb - np.array(empty_color, np.int16)).max(axis=2) > tolerance
        used += int(in_use.sum())
        black += int((in_use & (rgb.max(axis=2) < black_level)).sum())
    return {"images": len(images), "used_texels": used,
            "black_pct": round(100.0 * black / used, 2) if used else None}
