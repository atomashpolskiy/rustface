#!/usr/bin/env sh

if [[ $RUSTFACE_HOME == "" ]]
then
    export RUSTFACE_HOME=$PWD
fi
echo "Using ${RUSTFACE_HOME} as the working directory"

cargo run --release --features opencv-demo \
    ${RUSTFACE_HOME}/model/seeta_fd_frontal_v1.0.bin \
    ${RUSTFACE_HOME}/assets/test/scientists.jpg
