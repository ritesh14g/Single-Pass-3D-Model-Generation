"""Stage 5 (Georeferencing & export, §8.1–8.3) scorecard.

Every file is re-opened, not trusted from the writer: the §11 definition of done asks for
"all six output formats with valid, verified georeferencing".

  * Georeferencing: referenced, CRS = the flight's UTM zone, orthometric heights, camera
    centres vs GPS (the PS's <= 1 m), RANSAC inlier share.
  * Formats: each of OBJ / PLY / LAS / GeoTIFF / glb / FBX present and readable; LAS and
    GeoTIFF CRS read back from the headers; confidence carried in PLY and LAS.
  * Coverage: share of the ground the cameras saw that the dense cloud covers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.qa.stage1_eval import FAIL, INFO, PASS, WARN, Kpi, StageEvaluation

DEFAULTS = {"cam_vs_gps_rms_pass_m": 1.0, "cam_vs_gps_rms_warn_m": 5.0, "inlier_fraction_pass": 0.9,
            "inlier_fraction_warn": 0.7, "coverage_pass_pct": 90.0, "coverage_warn_pct": 70.0,
            "gcp_check_rms_pass_m": 1.0, "gcp_check_rms_warn_m": 3.0}


@dataclass
class ExportOutputs:
    georef: dict[str, Any]
    metadata: dict[str, Any]
    export_dir: Path

    @classmethod
    def load(cls, run_dir: Path | str) -> "ExportOutputs":
        run_dir = Path(run_dir)

        def rj(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

        return cls(rj(run_dir / "geo" / "georef.json"), rj(run_dir / "export" / "metadata.json"), run_dir / "export")


def _band(cfg: Any, key: str) -> float:
    try:
        value = cfg.get_path(f"qa.stage5.{key}")
    except Exception:
        value = None
    return float(DEFAULTS[key] if value is None else value)


def verify_file(fmt: str, path: Path) -> tuple[bool, str]:
    """Re-open one output; (ok, detail)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        if fmt == "las":
            import laspy

            with laspy.open(str(path)) as reader:
                crs = reader.header.parse_crs()
                dims = set(reader.header.point_format.dimension_names)
                return crs is not None, (f"{reader.header.point_count:,} points, CRS {crs.name if crs else 'MISSING'}"
                                         + ("" if "confidence" in dims else ", no confidence"))
        if fmt.startswith("geotiff"):
            import rasterio

            with rasterio.open(path) as src:
                return src.crs is not None, f"{src.width}x{src.height}, CRS {src.crs.to_string() if src.crs else 'MISSING'}"
        if fmt == "ply":
            from src.recon.meshing import ply_counts

            header = path.read_bytes()[:2048].decode("ascii", errors="ignore")
            return "confidence" in header, f"{ply_counts(path).get('vertices', 0):,} points" + (
                "" if "confidence" in header else ", no confidence property")
        if fmt.startswith("glb"):
            import pygltflib

            gltf = pygltflib.GLTF2().load(str(path))
            extras = gltf.extras or gltf.asset.extras or {}
            return bool(gltf.meshes), f"{len(gltf.meshes)} mesh(es), offset {extras.get('offset')}"
        if fmt == "obj":
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                head = fh.read(4096)
            mtl = next((ln.split(maxsplit=1)[1].strip() for ln in head.splitlines() if ln.startswith("mtllib ")), None)
            ok = "\nv " in head or head.startswith("v ")
            return ok, "textured" if mtl and (path.parent / mtl).exists() else "no material"
        if fmt == "fbx":
            return path.read_bytes()[:18] == b"Kaydara FBX Binary", f"{path.stat().st_size / 1e6:.1f} MB"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {type(exc).__name__}: {exc}"
    return True, ""


def evaluate_export(outputs: ExportOutputs, cfg: Any) -> StageEvaluation:
    ev = StageEvaluation(stage="geo_export")
    g, meta = outputs.georef, outputs.metadata
    grp = "Georeferencing"
    if not g:
        ev.kpis.append(Kpi("georef", grp, "georef.json", None, "present", FAIL, "geo stage did not run"))
        return ev
    frame = g.get("frame")
    ev.kpis.append(Kpi("referenced", grp, "Georeferenced", bool(frame), "yes", PASS if frame else FAIL,
                       g.get("reason", "") if not frame else f"{g.get('gps_frames')} GPS frames"))
    if frame:
        ev.kpis.append(Kpi("crs", grp, "Coordinate reference system", frame["crs"], "UTM zone of the flight", PASS,
                           "WKT in LAS, EPSG in GeoTIFF, offset in OBJ/glb/PLY sidecars"))
        ortho = frame["vertical_epsg"] is not None or frame["gps_altitude_datum"] == "orthometric"
        ev.kpis.append(Kpi("vertical_datum", grp, "Heights", frame["vertical_datum"], "orthometric",
                           PASS if ortho else WARN, "; ".join(frame.get("notes") or []) + _datum_detail(frame, g)))
        rms = float(g.get("rms_all_m", float("nan")))
        ev.kpis.append(Kpi("cam_vs_gps_rms_m", grp, "Camera centres vs GPS (RMS, all)", rms,
                           f"<= {_band(cfg, 'cam_vs_gps_rms_pass_m')} m (PS spatial accuracy)",
                           PASS if rms <= _band(cfg, "cam_vs_gps_rms_pass_m") else
                           WARN if rms <= _band(cfg, "cam_vs_gps_rms_warn_m") else FAIL,
                           f"horizontal {g.get('horizontal_rms_m')} m, vertical {g.get('vertical_rms_m')} m; "
                           f"inliers-only {g.get('rms_inliers_m')} m", unit="m"))
        share = g.get("inliers", 0) / max(g.get("cameras") or 1, 1)
        ev.kpis.append(Kpi("inlier_fraction", grp, "GPS fixes kept by RANSAC", round(share, 3),
                           f">= {_band(cfg, 'inlier_fraction_pass'):.2f}",
                           PASS if share >= _band(cfg, "inlier_fraction_pass") else
                           WARN if share >= _band(cfg, "inlier_fraction_warn") else FAIL,
                           f"{g.get('inliers')} of {g.get('cameras')} within {g.get('inlier_threshold_m')} m"))
        ev.kpis.append(Kpi("ground_constraint", grp, "Straight-path roll constraint", bool(g.get("ground_constraint")),
                           "-", INFO, f"track collinearity {g.get('track_collinearity')} (< 0.1 turns it on)"))
        gcp = g.get("gcp")
        if gcp:
            # S5-3. Check points are never fitted: their residual is accuracy against surveyed truth.
            notes = "; ".join(gcp.get("notes") or [])
            if gcp.get("check_rms_m") is not None:
                rms_chk = float(gcp["check_rms_m"])
                ev.kpis.append(Kpi("gcp_check_rms_m", grp, "GCP check points (independent accuracy)", rms_chk,
                                   f"<= {_band(cfg, 'gcp_check_rms_pass_m')}",
                                   PASS if rms_chk <= _band(cfg, "gcp_check_rms_pass_m") else
                                   WARN if rms_chk <= _band(cfg, "gcp_check_rms_warn_m") else FAIL,
                                   f"{gcp['check']} check points: horizontal {gcp.get('check_horizontal_rms_m')} m, "
                                   f"vertical {gcp.get('check_vertical_rms_m')} m" + (f"; {notes}" if notes else ""),
                                   unit="m"))
            ev.kpis.append(Kpi("gcp_control", grp, "Ground control points used", int(gcp.get("control") or 0), "-",
                               INFO if gcp.get("control") else WARN,
                               f"control RMS {gcp.get('control_rms_m')} m after the fit" + (f"; {notes}" if notes else "")))

    grp = "Formats (re-opened)"
    files = {f["format"]: f for f in meta.get("files", [])}
    failed = meta.get("formats_failed", {})
    for fmt, key in (("obj", "obj"), ("ply", "ply"), ("las", "las"), ("geotiff_dsm", "geotiff_dsm"),
                     ("geotiff_ortho", "geotiff_ortho"), ("glb", "glb"), ("fbx", "fbx")):
        entry = files.get(key)
        if entry is None:
            reason = failed.get(fmt.split("_")[0], "not written")
            ev.kpis.append(Kpi(f"format_{fmt}", grp, fmt.upper().replace("_", " "), False, "written + readable",
                               FAIL, reason))
            continue
        ok, detail = verify_file(fmt, Path(entry["path"]))
        ev.kpis.append(Kpi(f"format_{fmt}", grp, fmt.upper().replace("_", " "), ok, "written + readable",
                           PASS if ok else FAIL, f"{entry['bytes'] / 1e6:.1f} MB; {detail}"))
    if "glb_confidence" in files:
        ok, detail = verify_file("glb_confidence", Path(files["glb_confidence"]["path"]))
        ev.kpis.append(Kpi("confidence_layer", grp, "Confidence layer (glb)", ok, "written", PASS if ok else WARN, detail))
    if meta.get("zones"):  # Stage 3 ran: its layers must have reached the deliverables
        ok, detail = (verify_file("glb_zones", Path(files["glb_zones"]["path"])) if "glb_zones" in files
                      else (False, failed.get("glb_zones", "not written")))
        layers = (files.get("las") or {}).get("layers") or []
        ok = ok and "zone" in layers and "geojson_gaps" in files
        ev.kpis.append(Kpi("zones_layer", grp, "Stage 3 layers (zones glb, LAS/PLY zone, gaps.geojson)", ok,
                           "written", PASS if ok else WARN,
                           f"{detail}; LAS layers {layers}; gaps.geojson {'yes' if 'geojson_gaps' in files else 'no'}"))

    if "las" in files:
        # S5-3: read the classes back from the file, not from the export's own report.
        import laspy

        try:
            codes = np.asarray(laspy.read(str(files["las"]["path"])).classification)
            share = {c: float((codes == c).mean()) for c in (1, 2, 7)}
            classified = share[2] > 0
            ev.kpis.append(Kpi("las_classified", grp, "LAS classified (ground / above / noise)", classified,
                               "ground class present", PASS if classified else WARN,
                               f"ground {share[2]:.1%}, above ground {share[1]:.1%}, low noise {share[7]:.1%} "
                               "(ASPRS 2 / 1 / 7)"))
        except Exception as exc:  # noqa: BLE001
            ev.kpis.append(Kpi("las_classified", grp, "LAS classified (ground / above / noise)", False,
                               "ground class present", WARN, f"{type(exc).__name__}: {exc}"))

    grp = "Coverage"
    cov = meta.get("coverage")
    if cov:
        pct = float(cov["coverage_pct"])
        ev.kpis.append(Kpi("coverage_pct", grp, "Visible ground reconstructed", pct,
                           f">= {_band(cfg, 'coverage_pass_pct'):.0f}%",
                           PASS if pct >= _band(cfg, "coverage_pass_pct") else
                           WARN if pct >= _band(cfg, "coverage_warn_pct") else FAIL,
                           f"{cov['covered_m2']:,} of {cov['visible_m2']:,} m² the cameras saw", unit="%"))
    else:
        ev.kpis.append(Kpi("coverage_pct", grp, "Visible ground reconstructed", None, ">= 90%", INFO,
                           "not measured (no georeferencing or no dense cloud)"))
    dsm = files.get("geotiff_dsm")
    if dsm:
        ev.kpis.append(Kpi("dsm", grp, "DSM", f"{dsm.get('resolution_m')} m", "-", INFO,
                           f"{dsm.get('width')}x{dsm.get('height')}, {100 * dsm.get('valid_fraction', 0):.0f}% of cells "
                           f"valid ({dsm.get('filled_cells', 0):,} gap-filled)"))
    return ev


def residuals_frame(outputs: ExportOutputs):
    import pandas as pd

    return pd.DataFrame(outputs.georef.get("per_camera", []))


def save_evaluation(evaluation: StageEvaluation, path: Path | str) -> Path:
    path = Path(path)
    path.write_text(json.dumps(evaluation.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def _datum_detail(frame: dict, g: dict) -> str:
    """How the telemetry altitude's datum was decided (S5-1): measured against terrain, or assumed."""
    check = g.get("altitude_datum") or {}
    datum = frame.get("gps_altitude_datum")
    if frame.get("gps_altitude_datum_assumed"):
        why = check.get("detail") or "no datum check ran"
        return f" GPS altitude datum ASSUMED {datum} ({why})."
    if check.get("method") in ("terrain check", "terrain at takeoff"):
        return f" GPS altitude datum measured {datum}: {check.get('detail')}."
    return f" GPS altitude datum {datum} (source or config)."
