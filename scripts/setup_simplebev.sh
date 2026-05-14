#!/usr/bin/env bash
# Clone Simple-BEV and fetch the camera-only checkpoint into external/simple_bev.
set -euo pipefail

EXT_DIR="external/simple_bev"
REPO_URL="https://github.com/aharley/simple_bev.git"

mkdir -p external
if [ ! -d "${EXT_DIR}/.git" ]; then
    git clone --depth 1 "${REPO_URL}" "${EXT_DIR}"
fi

cd "${EXT_DIR}"
if compgen -G "checkpoints/8x5*" > /dev/null || compgen -G "8x5*" > /dev/null; then
    echo "rgb checkpoint already present"
else
    if [ ! -f "rgb_checkpoint.tar.gz" ]; then
        wget --no-check-certificate -O rgb_checkpoint.tar.gz \
            "https://www.dropbox.com/s/n93ryvrqyiram56/rgb_checkpoint.tar.gz?dl=1"
    fi
    tar -xvf rgb_checkpoint.tar.gz
    rm -v rgb_checkpoint.tar.gz
fi
