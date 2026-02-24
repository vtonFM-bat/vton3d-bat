#!/bin/bash
set -euo pipefail

SBATCH_SCRIPT="./run_pipeline.sh"
BASE_YAML="configs/vton_pipeline.yaml"

# Ordner vorbereiten
mkdir -p configs/prompt_engi_yamls logs

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

# Per-cloth prompts (used to build the prompts mapping in the YAML)
declare -A PROMPT=(
  [shirt]="Replace ONLY the person's original shirt with the clothing image. Do not redesign or stylize the garment. The new garment must match the reference exactly: same colors, fabric texture, pattern/print, logos, seams/stitching, neckline/collar, sleeve length, fit, and hem/waist details. Keep background, pose, face, hair, skin, body shape, all other clothes of the person, lighting and camera exactly unchanged. Ensure temporal consistency. Produce clean boundaries where the new garment meets skin/other clothing; remove all remnants of the original shirt."
  [pant]="Replace ONLY the person's original pant with the clothing image. Do not redesign or stylize the garment. The new garment must match the reference exactly: same colors, fabric texture, pattern/print, logos, seams/stitching, neckline/collar, sleeve length, fit, and hem/waist details. Keep background, pose, face, hair, skin, body shape, all other clothes of the person, lighting and camera exactly unchanged. Ensure temporal consistency. Produce clean boundaries where the new garment meets skin/other clothing; remove all remnants of the original pant."
  [dress]="Replace ONLY the person's original pant and shirt with the clothing image. Do not redesign or stylize the garment. The new garment must match the reference exactly: same colors, fabric texture, pattern/print, logos, seams/stitching, neckline/collar, sleeve length, fit, and hem/waist details. Keep background, pose, face, hair, skin, body shape, all other clothes of the person, lighting and camera exactly unchanged. Ensure temporal consistency. Produce clean boundaries where the new garment meets skin/other clothing; remove all remnants of the original pant and shirt."
)

declare -A NEG_PROMPT=(
  [shirt]="pose change, body shape change, face change, hair change, background change, lighting change, camera change, style transfer, redesign, different garment, different color, different fabric, different pattern, different logo, extra clothing pieces, text, watermark, artifacts, blended old clothing, visible remnants of original clothing."
  [pant]="pose change, body shape change, face change, hair change, background change, lighting change, camera change, style transfer, redesign, different garment, different color, different fabric, different pattern, different logo, extra clothing pieces, text, watermark, artifacts, blended old clothing, visible remnants of original clothing."
  [dress]="pose change, body shape change, face change, hair change, background change, lighting change, camera change, style transfer, redesign, different garment, different color, different fabric, different pattern, different logo, extra clothing pieces, text, watermark, artifacts, blended old clothing, visible remnants of original clothing."
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
    run_name="Exp00_${person}_${cloth}"
    video_name="${VIDEO[$person]}"
    clothing_image="${CLOTHING_PATH[$cloth]}"

    out_cfg="configs/prompt_engi_yamls/${run_name}.yaml"

    # Base-YAML kopieren
    cp "$BASE_YAML" "$out_cfg"

    # prompt + negative_prompt je nach Kleidungsstück aus den Arrays
    prompt="${PROMPT[$cloth]}"
    neg_prompt="${NEG_PROMPT[$cloth]}"


    # YAML-Felder ersetzen
    sed -i "s|^  scene_dir: .*|  scene_dir: \"${scene_dir}\"|g" "$out_cfg"
    sed -i "s|^  runs_root: .*|  runs_root: \"data/train/${person}/sweep_runs\"   # optional|g" "$out_cfg"
    sed -i "s|^  run_name: .*|  run_name: \"${run_name}\"|g" "$out_cfg"
    sed -i "s|^  video_name: .*|  video_name: \"${video_name}\"|g" "$out_cfg"
    sed -i "s|^  clothing_image: .*|  clothing_image: \"${clothing_image}\"|g" "$out_cfg"
    sed -i "/^qwen:$/,/^[a-zA-Z_][a-zA-Z0-9_]*:$/ s|^  prompt: .*|  prompt: \"${prompt}\"|g" "$out_cfg"
    sed -i "/^qwen:$/,/^[a-zA-Z_][a-zA-Z0-9_]*:$/ s|^  negative_prompt: .*|  negative_prompt: \"${neg_prompt}\"|g" "$out_cfg"
    sed -i "/^qwen:$/,/^[a-zA-Z_][a-zA-Z0-9_]*:$/{
/^  negative_prompt:/a\\
  prompts:\\
    upper: \"${PROMPT[shirt]}\"\\
    lower: \"${PROMPT[pant]}\"\\
    dress: \"${PROMPT[dress]}\"\\
  negative_prompts:\\
    upper: \"${NEG_PROMPT[shirt]}\"\\
    lower: \"${NEG_PROMPT[pant]}\"\\
    dress: \"${NEG_PROMPT[dress]}\"
}" "$out_cfg"


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
