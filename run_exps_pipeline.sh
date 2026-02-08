#!/bin/bash
set -euo pipefail

SBATCH_SCRIPT="./run_pipeline.sh"
BASE_YAML="configs/vton_pipeline.yaml"

# Ordner vorbereiten
mkdir -p configs/optical_flow_yamls logs

# Person -> Video
declare -A VIDEO=(
  [florian]="florian_shirt.MOV"
  [can]="can_jaeckli.MOV"
  [jan]="jan_fs.MOV"
  [petra]="petra_fs.MOV"
)

# Clothing Varianten
clothes=("shirt" "pant" "dress")

declare -A CLOTHING_PATH=(
  [shirt]="data/clothing/train/upper/short/shirts/white_plants_shirt.jpg"
  [pant]="data/clothing/train/lower/long/pants/gray_pant.jpg"
  [dress]="data/clothing/train/dress/black_dress.jpg"
)

# Seriell laufen lassen (Dependency-Kette)
SERIAL=1
prev_jobid=""

for person in florian can jan petra; do
  for cloth in "${clothes[@]}"; do

    # dress nur für petra
    if [[ "$cloth" == "dress" && "$person" != "petra" ]]; then
      continue
    fi

    scene_dir="data/train/${person}/"
    run_name="vton_opflow_${person}_${cloth}"
    video_name="${VIDEO[$person]}"
    clothing_image="${CLOTHING_PATH[$cloth]}"

    out_cfg="configs/prompt_engi_yamls/${run_name}.yaml"

    # Base-YAML kopieren
    cp "$BASE_YAML" "$out_cfg"

    # negative_prompt dynamisch: alle anderen Kleidungsstücke (zusätzlich zu "change pose")
    neg_prompt="change pose"
    for c in "${clothes[@]}"; do
      [[ "$c" == "$cloth" ]] && continue
      neg_prompt+=", change ${c}"
    done

    # YAML-Felder ersetzen
    sed -i "s|^  scene_dir: .*|  scene_dir: \"${scene_dir}\"|g" "$out_cfg"
    sed -i "s|^  runs_root: .*|  runs_root: \"data/train/${person}/sweep_runs\"   # optional|g" "$out_cfg"
    sed -i "s|^  run_name: .*|  run_name: \"${run_name}\"|g" "$out_cfg"
    sed -i "s|^  video_name: .*|  video_name: \"${video_name}\"|g" "$out_cfg"
    sed -i "s|^  clothing_image: .*|  clothing_image: \"${clothing_image}\"|g" "$out_cfg"
    sed -i "s|^  negative_prompt: .*|  negative_prompt: \"${neg_prompt}\"|g" "$out_cfg"

    echo "Submitting ${run_name}"
    echo "  config: $out_cfg"
    echo "  negative_prompt: $neg_prompt"

    if [[ "$SERIAL" -eq 1 ]]; then
      if [[ -z "$prev_jobid" ]]; then
        jobid=$(sbatch --parsable "$SBATCH_SCRIPT" "$out_cfg")
      else
        jobid=$(sbatch --parsable --dependency=afterok:"$prev_jobid" "$SBATCH_SCRIPT" "$out_cfg")
      fi
      prev_jobid="$jobid"
      echo " -> jobid: $jobid (serial)"
    else
      jobid=$(sbatch --parsable "$SBATCH_SCRIPT" "$out_cfg")
      echo " -> jobid: $jobid (parallel)"
    fi

    echo ""
  done
done

echo "All jobs submitted."
