"""Merge disconnected SfM models through GPS.

When SfM splits a flight into pieces (on the box, Esri: 42 of 50 frames in the largest of
two models, after Stage 1 widened the frame spacing to meet its budget), the smaller pieces
used to be dropped. They share no images with the main piece, so COLMAP cannot merge them
by image correspondence; they can be placed through GPS instead, because every piece can be
fitted to GPS on its own. A piece is moved into the main piece's frame by
``main_from_gps · gps_from_piece`` and its cameras and points are copied in.

The join is only as good as the two GPS fits, so each piece's residual is reported: a large
one means the seam is off by about that much.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.recon import alignment


def posed(rec) -> list[Any]:
    """Images with a pose. A model can also hold unregistered images, and asking one of
    those for its projection centre aborts inside COLMAP."""
    return [im for im in rec.images.values() if im.has_pose]


def _gps_fit(rec, gps: dict[str, np.ndarray]):
    names = sorted(im.name for im in posed(rec))
    by_name = {im.name: im for im in posed(rec)}
    centres = np.array([by_name[n].projection_center() for n in names])
    return alignment.metric_check(names, centres, np.empty((0, 3)), gps)


def merge_by_gps(models: list[Any], gps: dict[str, np.ndarray], min_frames: int = 3,
                 max_rms_m: float = 25.0) -> tuple[Any, dict[str, Any]]:
    """Largest model with every GPS-placeable smaller model merged into it.

    Returns ``(merged, info)``. Models with fewer than ``min_frames`` GPS-matched frames,
    or whose own GPS fit is worse than ``max_rms_m``, are left out and counted.
    """
    import pycolmap

    models = sorted(models, key=lambda m: m.num_reg_images(), reverse=True)
    main = models[0]
    info: dict[str, Any] = {"models": len(models), "merged": 0, "frames_added": 0, "skipped": [],
                            "piece_gps_rms_m": []}
    if len(models) == 1:
        return main, info
    main_stats, main_fit = _gps_fit(main, gps)
    if main_fit is None:
        info["skipped"] = [f"main model has {main_stats.get('gps_matched_frames', 0)} GPS frames"]
        return main, info
    info["main_gps_rms_m"] = main_stats["cam_vs_gps_rms_m"]
    s_p, r_p, t_p = main_fit
    merged = pycolmap.Reconstruction(main)
    camera_id = next(iter(merged.cameras))
    next_image_id = max(merged.images) + 1

    for piece in models[1:]:
        stats, fit = _gps_fit(piece, gps)
        if fit is None or stats["gps_matched_frames"] < min_frames:
            info["skipped"].append(f"{piece.num_reg_images()} frames: too few GPS fixes")
            continue
        if stats["cam_vs_gps_rms_m"] > max_rms_m:
            info["skipped"].append(f"{piece.num_reg_images()} frames: GPS fit {stats['cam_vs_gps_rms_m']} m")
            continue
        s_m, r_m, t_m = fit
        # x_main = main_from_gps(gps_from_piece(x)) = (s_m/s_p) R_p^T R_m x + R_p^T (t_m - t_p) / s_p
        moved = pycolmap.Reconstruction(piece)
        moved.transform(pycolmap.Sim3d(s_m / s_p, pycolmap.Rotation3d(r_p.T @ r_m), r_p.T @ (t_m - t_p) / s_p))

        id_map: dict[int, int] = {}
        for image in posed(moved):
            existing = merged.find_image_with_name(image.name)
            if existing is not None and existing.has_pose:
                continue  # registered in both pieces: keep the main piece's pose
            if existing is not None:
                # Held unregistered by the main model (same features, same database): pose it.
                frame = merged.frame(existing.frame_id)
                frame.set_cam_from_world(existing.camera_id, image.cam_from_world())
                merged.register_frame(existing.frame_id)
                id_map[image.image_id] = existing.image_id
                continue
            copy = pycolmap.Image(name=image.name,
                                  points2D=[pycolmap.Point2D(p.xy) for p in image.points2D],
                                  camera_id=camera_id, image_id=next_image_id)
            merged.add_image_with_trivial_frame(copy, image.cam_from_world())
            id_map[image.image_id] = next_image_id
            next_image_id += 1
        for point in moved.points3D.values():
            track = pycolmap.Track()
            for element in point.track.elements:
                if element.image_id in id_map:
                    track.add_element(id_map[element.image_id], element.point2D_idx)
            if track.length() >= 2:
                merged.add_point3D(point.xyz, track, point.color)
        info["merged"] += 1
        info["frames_added"] += len(id_map)
        info["piece_gps_rms_m"].append(stats["cam_vs_gps_rms_m"])
    return merged, info
