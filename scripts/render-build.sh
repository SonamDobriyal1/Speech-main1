#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FRONTEND_DIR="$ROOT_DIR/sambhav_technovation-main/frontend"
OUTPUT_DIR="$ROOT_DIR/static/voice-lab"

echo "==> Installing Python dependencies"
pip install -r requirements.txt

ensure_node() {
  if command -v npm >/dev/null 2>&1; then
    echo "==> Using system npm ($(npm --version))"
    return 0
  fi

  NODE_VERSION="${NODE_VERSION:-20.18.0}"
  NODE_ARCH="linux-x64"
  NODE_DIR="/tmp/node-v${NODE_VERSION}-${NODE_ARCH}"

  if [ ! -x "$NODE_DIR/bin/npm" ]; then
    echo "==> Installing Node.js ${NODE_VERSION} for Phoneme Coach build"
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-${NODE_ARCH}.tar.xz" \
      | tar -xJ -C /tmp
  fi

  export PATH="$NODE_DIR/bin:$PATH"
  echo "==> Using Node $(node --version), npm $(npm --version)"
}

build_frontend() {
  if [ ! -d "$FRONTEND_DIR" ]; then
    echo "ERROR: Frontend directory not found: $FRONTEND_DIR"
    exit 1
  fi

  ensure_node

  echo "==> Building Phoneme Coach frontend"
  cd "$FRONTEND_DIR"

  # npm ci is strict about lockfile/platform parity; npm install is safer on Render Linux.
  if ! npm install --no-audit --no-fund; then
    cd "$ROOT_DIR"
    if [ -f "$OUTPUT_DIR/index.html" ]; then
      echo "WARN: npm install failed; using committed Phoneme Coach assets in $OUTPUT_DIR"
      return 0
    fi
    echo "ERROR: npm install failed and no committed Phoneme Coach assets were found"
    exit 1
  fi

  npm run build
  cd "$ROOT_DIR"
}

if [ -f "$OUTPUT_DIR/index.html" ] && [ "${SKIP_VOICE_LAB_BUILD:-}" = "1" ]; then
  echo "==> Skipping frontend build (SKIP_VOICE_LAB_BUILD=1)"
elif [ -f "$OUTPUT_DIR/index.html" ] && [ "${FORCE_VOICE_LAB_BUILD:-}" != "1" ]; then
  echo "==> Reusing committed Phoneme Coach build in $OUTPUT_DIR"
  echo "    Set FORCE_VOICE_LAB_BUILD=1 to rebuild from source on deploy"
else
  build_frontend
fi

if [ ! -f "$OUTPUT_DIR/index.html" ]; then
  echo "ERROR: Missing Phoneme Coach build output at $OUTPUT_DIR/index.html"
  exit 1
fi

echo "==> Render build finished successfully"
