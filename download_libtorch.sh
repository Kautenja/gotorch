#!/bin/bash
# Download script for libtorch.
#
# Copyright (c) 2023 Christian Kauten
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXTERNRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Get the source directory for creating the build directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the operating system to determine which libtorch archive to download.
OS=$(uname | tr '[:upper:]' '[:lower:]')
# Get the architecture to detect x86 vs ARM builds.
ARCH=$(uname -m)
# Determine whether NVCC is installed on this system for a CUDA install.
NVCC=$(whereis cuda | cut -f 2 -d ' ')"/bin/nvcc"

function download_libtorch_x86_darwin() {
    (cd ${DIR}/build; curl --url "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.11.0.zip" --output "libtorch.zip")
}

function download_libtorch_arm64_darwin() {
    (cd ${DIR}/build; wget -O "libtorch.zip" 'https://github.com/Kautenja/libtorch-binaries/releases/download/v1.0.0/libtorch-shared-with-deps-arm64-darwin-1.11.0.zip')
}

function download_libtorch_x86_linux(){
    (cd ${DIR}/build; curl --url 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.11.0%2Bcpu.zip' --output "libtorch.zip")
}

function download_libtorch_armv7l_linux() {
    (cd ${DIR}/build; wget -O "libtorch.zip" 'https://github.com/Kautenja/libtorch-binaries/releases/download/v1.0.0/libtorch-shared-with-deps-armv7l-linux-1.11.0.zip')
}

function download_libtorch_aarch64_linux(){
    (cd ${DIR}/build; wget -O "libtorch.zip" 'https://github.com/Kautenja/libtorch-binaries/releases/download/v1.0.0/libtorch-shared-with-deps-aarch64-linux-1.11.0.zip')
}

function download_libtorch_cuda() {
    CUDA_VERSION=$("$NVCC" --version | grep release | grep -Eo "[0-9]+.[0-9]+" | head -1)
    if [[ "$CUDA_VERSION" == "10.1" ]]; then
        CUDA_VERSION="101"
    elif [[ "$CUDA_VERSION" == "10.2" ]]; then
        CUDA_VERSION="102"
    else
        echo "Unsupported CUDA version: $CUDA_VERSION"
        return -1
    fi
    (cd ${DIR}/build; curl -url 'https://download.pytorch.org/libtorch/cu$CUDA_VERSION/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu$CUDA_VERSION.zip' --output "libtorch.zip")
}

mkdir -p ${DIR}/build

echo "Downloading libtorch for $ARCH $OS to $DIR/build/libtorch"
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
else
    echo "Unsupported OS ($OS) and architecture ($ARCH) combination for libtorch"
    exit 1
fi
(cd ${DIR}/build; unzip -o libtorch.zip)
