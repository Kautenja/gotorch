# A docker file to build a CI image for GoTorch.
#
# Author: Christian Kauten (ckauten@sensoryinc.com)
#
# Copyright (c) 2021 Sensory, Inc.
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

# ----------------------------------------------------------------------------
# MARK: Image Setup
# ----------------------------------------------------------------------------

# Download base image Ubuntu 22.04.
FROM ubuntu:22.04
# Set the maintainer label for this Docker image.
LABEL MAINTAINER "Christian Kauten ckauten@sensoryinc.com"
# Set the time-zone of the machine.
ENV TZ=US/Mountain
# Disable the interactive dialogue for `dpkg` (running behind apt-get).
ARG DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# MARK: Dependencies
# ----------------------------------------------------------------------------

# Update Ubuntu software repositories.
RUN apt-get --fix-missing update
RUN apt-get -y upgrade
# Developer tools
RUN apt-get install -y htop nano vim git screen tmux bash
# Build chain for C/C++
RUN apt-get install -y build-essential clang autoconf libtool pkg-config curl wget zip cmake
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y golang-go
# Cleanup the package manager to remove unnecessary caches
RUN apt -y autoremove
RUN apt -y clean

# ----------------------------------------------------------------------------
# MARK: Project Files
# ----------------------------------------------------------------------------

RUN mkdir /gotorch
RUN mkdir /gotorch/cgotorch
# C and C++ header/source/build files.
COPY ./cgotorch/*.h /gotorch/cgotorch/
COPY ./cgotorch/*.hpp /gotorch/cgotorch/
COPY ./cgotorch/*.cc /gotorch/cgotorch/
COPY ./cgotorch/*.cpp /gotorch/cgotorch/
COPY ./CMakeLists.txt /gotorch/CMakeLists.txt
COPY ./install_libtorch.sh /gotorch/install_libtorch.sh
COPY ./download_libtorch.sh /gotorch/download_libtorch.sh
COPY ./build_libcgotorch.sh /gotorch/build_libcgotorch.sh
COPY ./install_libcgotorch.sh /gotorch/install_libcgotorch.sh
COPY ./install.sh /gotorch/install.sh
# GoLang files
COPY ./*.go /gotorch/
COPY ./go.* /gotorch/
COPY ./nn /gotorch/cuda
COPY ./nn /gotorch/nn
COPY ./jit /gotorch/jit
COPY ./internal /gotorch/internal
COPY ./vision /gotorch/vision
COPY ./cmd /gotorch/cmd
# Project files
COPY ./main.sh /gotorch/main.sh
# Test data/files
COPY ./data /gotorch/data

WORKDIR /gotorch
