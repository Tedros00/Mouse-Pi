#!/usr/bin/env bash
set -e

# Resolve project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Clean Windows Zone identifiers (harmless if none exist)
find "$ROOT" -name "*:Zone.Identifier" -delete || true

# Gazebo resource paths (models + worlds)
export GZ_SIM_RESOURCE_PATH="$ROOT/models:$ROOT/worlds:${GZ_SIM_RESOURCE_PATH}"

# Use default system plugins (DO NOT override unless you compiled custom ones)
unset GZ_SIM_SYSTEM_PLUGIN_PATH

echo "[INFO] Project root: $ROOT"
echo "[INFO] GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
echo "[INFO] Launching maze with robot..."

# Launch Gazebo Sim
gz sim -r "$ROOT/worlds/maze.sdf"
