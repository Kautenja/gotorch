# CMake build script for libcgotorch
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

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cgotorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET cgotorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cgotorch>)
endif (MSVC)

# --- Library ----------------------------------------------------------------

include_directories(.)
add_library(cgotorch SHARED
  cgotorch/byte_buffer.h
  cgotorch/cgotorch.h
  cgotorch/cuda.h
  cgotorch/device.h
  cgotorch/functional.h
  cgotorch/functions.h
  cgotorch/init.h
  cgotorch/ivalue.h
  cgotorch/jit.h
  cgotorch/memory.h
  cgotorch/optim.h
  cgotorch/tensor.h
  cgotorch/tensor_options.h
  cgotorch/torchdef.h
  cgotorch/try_catch_return_error_string.hpp
  cgotorch/vision_ops_nms.hpp
  cgotorch/vision_ops_roi_align.hpp
  cgotorch/vision_ops_roi_align_common.h
  cgotorch/byte_buffer.cc
  cgotorch/cuda.cc
  cgotorch/device.cc
  cgotorch/functional.cc
  cgotorch/functions.cc
  cgotorch/init.cc
  cgotorch/ivalue.cc
  cgotorch/jit.cc
  cgotorch/memory.cc
  cgotorch/optim.cc
  cgotorch/tensor.cc
  cgotorch/tensor_options.cpp
  cgotorch/torchdef.cc
  cgotorch/vision_ops_nms.cpp
  cgotorch/vision_ops_nms_kernel_cpu.cpp
  cgotorch/vision_ops_roi_align.cpp
  cgotorch/vision_ops_roi_align_kernel_cpu.cpp)
set_property(TARGET cgotorch PROPERTY CXX_STANDARD 14)
target_include_directories(cgotorch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cgotorch>
    $<INSTALL_INTERFACE:include>
    # cgotorch
)
set_target_properties(cgotorch PROPERTIES
    PUBLIC_HEADER cgotorch/cgotorch.hpp
)
target_link_libraries(cgotorch PUBLIC "${TORCH_LIBRARIES}")

# --- Install ----------------------------------------------------------------

install(TARGETS cgotorch
  EXPORT cgotorchTargets
  LIBRARY DESTINATION lib COMPONENT Runtime
  ARCHIVE DESTINATION lib COMPONENT Development
  RUNTIME DESTINATION bin COMPONENT Runtime
  PUBLIC_HEADER DESTINATION cgotorch/cgotorch.hpp COMPONENT Development
  BUNDLE DESTINATION bin COMPONENT Runtime
)
install(EXPORT cgotorchTargets DESTINATION lib/cmake/cgotorch)
install(FILES
    cgotorch/byte_buffer.h
    cgotorch/cgotorch.h
    cgotorch/cuda.h
    cgotorch/device.h
    cgotorch/functional.h
    cgotorch/functions.h
    cgotorch/init.h
    cgotorch/ivalue.h
    cgotorch/jit.h
    cgotorch/memory.h
    cgotorch/optim.h
    cgotorch/tensor.h
    cgotorch/tensor_options.h
    cgotorch/torchdef.h
    cgotorch/try_catch_return_error_string.hpp
    cgotorch/vision_ops_nms.hpp
    cgotorch/vision_ops_roi_align.hpp
    cgotorch/vision_ops_roi_align_common.h
  DESTINATION include
)
