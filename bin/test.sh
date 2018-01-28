#!/usr/bin/env sh

IMAGE_PATH=${1}
if [[ ${IMAGE_PATH} == "" ]]
then
    echo "Usage: ${0} <path-to-image>"
    exit 1
fi

if [[ ${RUSTFACE_HOME} == "" ]]
then
    export RUSTFACE_HOME=$PWD
fi
echo "Using ${RUSTFACE_HOME} as the working directory"

export RAYON_NUM_THREADS=2
cargo run --release --example image_demo \
    ${RUSTFACE_HOME}/model/seeta_fd_frontal_v1.0.bin \
    ${IMAGE_PATH}
