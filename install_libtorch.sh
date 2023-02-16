#!/bin/bash
# Installation script for libtorch.
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
# Ensure the output directories exits
$(which sudo) mkdir -p /usr/local/bin
$(which sudo) mkdir -p /usr/local/include
$(which sudo) mkdir -p /usr/local/lib
$(which sudo) mkdir -p /usr/local/share
# Copy the build artifacts to the system
(cd ${DIR}/build; $(which sudo) cp -r libtorch/* /usr/local/)
# Reload on Linux
if [[ $(uname) == "Linux" ]]; then
    ldconfig
fi
