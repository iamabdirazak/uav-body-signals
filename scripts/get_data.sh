#!/usr/bin/env bash
# Download the UAV-GESTURE keypoints and splits (about 16 MB) into data/.
# The keypoints are OpenPose estimates released by the dataset authors with their code:
# https://github.com/asankagp/UAV-GESTURE. The videos themselves are available on request
# (https://asankagp.github.io/uavgesture) and are not needed here.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ -d data/UAV-GESTURE/UAVGESTURE/joint_positions ]; then
  echo "Already downloaded: data/UAV-GESTURE"
  exit 0
fi
git clone --depth 1 https://github.com/asankagp/UAV-GESTURE data/UAV-GESTURE
echo "Keypoints ready in data/UAV-GESTURE/UAVGESTURE"
