#!/bin/bash

# Get the source directory for creating the build directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd $DIR > /dev/null

# Get the operating system to determine which libtorch archive to download.
OS=$(uname | tr '[:upper:]' '[:lower:]')
# Get the architecture to detect x86 vs ARM builds.
ARCH=$(uname -m)
# Determine whether NVCC is installed on this system for a CUDA install.
NVCC=$(whereis cuda | cut -f 2 -d ' ')"/bin/nvcc"
# Determine the number of logical compute cores on the CPU.
if [[ "$OS" == "darwin" ]]; then
    LOGICAL_CORES=$(sysctl -n hw.logicalcpu)
else
    LOGICAL_CORES=$(nproc --ignore=1)
fi

function download_libtorch_x86_darwin() {
    if [[ ! -d "$DIR/build/libtorch" ]]; then
        (cd build; curl --url "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.11.0.zip" --output "libtorch.zip")
    fi
}

function download_libtorch_arm64_darwin() {
    if [[ ! -d "$DIR/build/libtorch" ]]; then
        (cd build; wget -O "libtorch.zip" 'https://github.com/Kautenja/libtorch-binaries/releases/download/v1.0.0/libtorch-shared-with-deps-arm64-darwin-1.11.0.zip')
    fi
}

function download_libtorch_x86_linux(){
    if [[ ! -d "$DIR/build/libtorch" ]]; then
        (cd build; curl --url 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.11.0%2Bcpu.zip' --output "libtorch.zip")
    fi
}

function download_libtorch_armv7l_linux() {
    if [[ ! -d "$DIR/build/libtorch" ]]; then
        (cd build; wget -O "libtorch.zip" 'https://github.com/Kautenja/libtorch-binaries/releases/download/v1.0.0/libtorch-shared-with-deps-armv7l-linux-1.11.0.zip')
    fi
}

function download_libtorch_aarch64_linux(){
    if [[ ! -d "$DIR/build/libtorch" ]]; then
        (cd build; wget -O "libtorch.zip" 'https://github.com/Kautenja/libtorch-binaries/releases/download/v1.0.0/libtorch-shared-with-deps-aarch64-linux-1.11.0.zip')
    fi
}

function download_libtorch_cuda() {
    CUDA_VERSION=$("$NVCC" --version | grep release | grep -Eo "[0-9]+.[0-9]+" | head -1)
    if [[ "$CUDA_VERSION" == "10.1" ]]; then
        CUDA_VERSION="101"
    elif [[ "$CUDA_VERSION" == "10.2" ]]; then
        CUDA_VERSION="102"
    else
        echo "Not supported CUDA version. $CUDA_VERSION"
        return -1
    fi
    if [[ ! -d "$DIR/build/libtorch" ]]; then
        (cd build; curl -url 'https://download.pytorch.org/libtorch/cu$CUDA_VERSION/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu$CUDA_VERSION.zip' --output "libtorch.zip")
    fi
}

mkdir -p build

echo "Building CGoTorch for $ARCH $OS with $LOGICAL_CORES threads"
if [[ "$OS" == "darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        download_libtorch_arm64_darwin
    else
        download_libtorch_x86_darwin
    fi
elif [[ "$OS" == "linux" ]]; then
    if [[ "$ARCH" == "aarch64" ]]; then
        download_libtorch_aarch64_linux
    elif [[ "$ARCH" =~ arm* ]]; then
        download_libtorch_armv7l_linux
    elif "$NVCC" --version > /dev/null; then
        if ! download_libtorch_cuda; then
            download_libtorch_x86_linux
        fi
    else
        download_libtorch_x86_linux
    fi
fi

(cd build; unzip -qq -o libtorch.zip)
(cd build; cmake -DCMAKE_PREFIX_PATH=$(pwd)/libtorch ..)
(cd build; make -j${LOGICAL_CORES})

popd > /dev/null
