#!/bin/bash
# Build script for libcgotorch.
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
set -x
set -e
set -o pipefail
# Get the source directory for creating the build directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the operating system to determine which libtorch archive to download.
OS=$(uname | tr '[:upper:]' '[:lower:]')
# Get the architecture to detect x86 vs ARM builds.
ARCH=$(uname -m)
# Determine the number of logical compute cores on the CPU.
if [[ "$OS" == "darwin" ]]; then
    LOGICAL_CORES=$(sysctl -n hw.logicalcpu)
else
    LOGICAL_CORES=$(nproc --ignore=1)
fi
# Make the build directory
mkdir -p ${DIR}/build
# Build the library
echo "Building libcgotorch for $ARCH $OS with $LOGICAL_CORES threads"
(cd ${DIR}/build; cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:PATH=libcgotorch ..)
(cd ${DIR}/build; make -j${LOGICAL_CORES} install)
