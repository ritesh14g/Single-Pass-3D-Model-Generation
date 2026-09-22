#!/usr/bin/env bash
# Stages 1-4 on the GPU box with VGGT-Omega and with VGGT-1B, from one shared Stage 1-2 run,
# plus Omega's depth accuracy against COLMAP (hybrid probe). One report at the end.
#
#   bash scripts/box_stage4_compare.sh [video] [tag]
#
# Needs tools/vggt, tools/vggt_omega and a Hugging Face login with VGGT-Omega access.
set -uo pipefail
VIDEO=${1:-data/raw/Esri_multiplexer_1.mp4}
TAG=${2:-esri}
PY=${PY:-.venv/bin/python}
BASE=data/interim/${TAG}_base

echo "== 1/5 Stages 1-2 once ($VIDEO)"
$PY -m src.cli run "$VIDEO" --out "$BASE" --stage ingest --stage condition > "${TAG}_stages12.log" 2>&1 \
  || { echo "Stages 1-2 failed: see ${TAG}_stages12.log"; exit 1; }

for model in vggt_omega vggt; do
  n=$([ "$model" = vggt_omega ] && echo 2 || echo 3)
  echo "== $n/5 Stage 4 with $model"
  rm -rf "data/interim/${TAG}_${model}" && cp -r "$BASE" "data/interim/${TAG}_${model}"
  $PY -m src.cli run "$VIDEO" --resume "data/interim/${TAG}_${model}" --stage track_a \
      --set recon.track_b.model=$model > "${TAG}_${model}.log" 2>&1 || echo "   Stage 4 with $model failed: see ${TAG}_${model}.log"
done

PROBE=""
if ls data/outputs/recon_probe/esri_gpu/dense/stereo/depth_maps/*.geometric.bin > /dev/null 2>&1 && [ "$TAG" = esri ]; then
  echo "== 4/5 VGGT-Omega depth vs COLMAP (hybrid probe)"
  $PY scripts/vggt_hybrid_probe.py --run-dir data/interim/esri_gpu --colmap-dense data/outputs/recon_probe/esri_gpu/dense \
      --out data/outputs/vggt_probe/esri_omega --model vggt_omega --widths 512 > "${TAG}_omega_probe.log" 2>&1 \
      && PROBE=data/outputs/vggt_probe/esri_omega/hybrid_summary.json || echo "   probe failed: see ${TAG}_omega_probe.log"
fi

echo "== 5/5 report"
$PY scripts/stage4_report.py "data/interim/${TAG}_vggt_omega" "data/interim/${TAG}_vggt" $PROBE
