#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="pretrain_human_perception_classifier_pp/models/human_perception_place_pulse"
ATTRIBUTE="safety"
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download_vitpp2.sh [--attribute safety|lively|wealthy|beautiful|boring|depressing|all] [--output-dir DIR] [--force]

Examples:
  scripts/download_vitpp2.sh
  scripts/download_vitpp2.sh --attribute safety
  scripts/download_vitpp2.sh --attribute all --output-dir pretrain_human_perception_classifier_pp/models/human_perception_place_pulse
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attribute)
      ATTRIBUTE="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

checkpoint_name() {
  case "$1" in
    safety|lively|wealthy|beautiful|boring|depressing)
      printf '%s.pth\n' "$1"
      ;;
    *)
      echo "Unsupported attribute: $1" >&2
      exit 1
      ;;
  esac
}

download_one() {
  local attribute="$1"
  local filename
  local url
  local target
  local partial

  filename="$(checkpoint_name "$attribute")"
  url="https://huggingface.co/Jiani11/human-perception-place-pulse/resolve/main/${filename}"
  target="${OUTPUT_DIR}/${filename}"
  partial="${target}.part"

  if [[ -f "$target" && "$FORCE" -ne 1 ]]; then
    echo "exists=${target}"
    return 0
  fi

  mkdir -p "$OUTPUT_DIR"
  if [[ "$FORCE" -eq 1 ]]; then
    rm -f "$target" "$partial"
  fi
  echo "download=${url}"
  curl -L --fail --progress-bar --continue-at - "$url" -o "$partial"
  mv "$partial" "$target"
  echo "saved=${target}"
}

if [[ "$ATTRIBUTE" == "all" ]]; then
  for name in safety lively wealthy beautiful boring depressing; do
    download_one "$name"
  done
else
  download_one "$ATTRIBUTE"
fi
