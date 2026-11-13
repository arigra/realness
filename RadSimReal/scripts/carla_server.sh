#!/usr/bin/env bash
# Start CARLA 0.9.15 headless in a given town with the user-space Vulkan loader. Usage: carla_server.sh <CARLA_DIR> <Town>
set -euo pipefail
V=/truenas/home/arigra/permuter/ariGranevich/carla/vulkan
export VK_ICD_FILENAMES=$V/nvidia_icd.json LD_LIBRARY_PATH=$V/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
cd "$1"
exec ./CarlaUE4.sh "/Game/Carla/Maps/$2" -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low
