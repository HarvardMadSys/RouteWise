#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Download RouteWise raw workload traces.

Usage:
  scripts/download_workloads.sh [--dataset all|burstgpt|sharegpt] [--force] [--no-verify]

Defaults:
  --dataset all

Environment overrides:
  BURSTGPT_URL   Override BurstGPT_3.csv source URL.
  SHAREGPT_URL   Override ShareGPT V3 source URL.

The script is idempotent: if the destination file already exists and passes
checksum verification, it is not downloaded again.
EOF
}

dataset="all"
force=0
verify=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "--dataset requires a value: all, burstgpt, or sharegpt" >&2
        exit 2
      fi
      dataset="${2:-}"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --no-verify)
      verify=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$dataset" in
  all|burstgpt|sharegpt) ;;
  *)
    echo "Invalid --dataset value: $dataset" >&2
    usage >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

burstgpt_dir="$repo_root/data/full/burstgpt"
sharegpt_dir="$repo_root/data/full/sharegpt"

burstgpt_path="$burstgpt_dir/BurstGPT_3.csv"
sharegpt_path="$sharegpt_dir/ShareGPT_V3_unfiltered_cleaned_split.json"

burstgpt_url="${BURSTGPT_URL:-https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv}"
sharegpt_url="${SHAREGPT_URL:-https://huggingface.co/datasets/learnanything/sharegpt_v3_unfiltered_cleaned_split/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"

burstgpt_sha256="2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"
sharegpt_sha256="35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "No sha256sum or shasum found for checksum verification." >&2
    return 1
  fi
}

verify_file() {
  local path="$1"
  local expected="$2"
  local actual

  [[ "$verify" -eq 1 ]] || return 0

  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
}

download_file() {
  local name="$1"
  local url="$2"
  local path="$3"
  local expected_sha256="$4"
  local tmp_path

  mkdir -p "$(dirname "$path")"

  if [[ -s "$path" && "$force" -eq 0 ]]; then
    if verify_file "$path" "$expected_sha256"; then
      echo "[skip] $name already exists: $path"
      return 0
    fi
    echo "[error] Existing $name file failed checksum verification." >&2
    echo "        Re-run with --force to replace it, or --no-verify to keep it." >&2
    return 1
  fi

  tmp_path="$path.part"
  echo "[download] $name"
  echo "           $url"
  echo "        -> $path"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --continue-at - --output "$tmp_path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$tmp_path" "$url"
  else
    echo "Need curl or wget to download workloads." >&2
    exit 1
  fi

  verify_file "$tmp_path" "$expected_sha256"
  mv "$tmp_path" "$path"
  echo "[ok] $name saved: $path"
}

if [[ "$dataset" == "all" || "$dataset" == "burstgpt" ]]; then
  download_file "BurstGPT_3" "$burstgpt_url" "$burstgpt_path" "$burstgpt_sha256"
fi

if [[ "$dataset" == "all" || "$dataset" == "sharegpt" ]]; then
  download_file "ShareGPT_V3" "$sharegpt_url" "$sharegpt_path" "$sharegpt_sha256"
fi
