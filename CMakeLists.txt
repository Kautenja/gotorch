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
file(GLOB_RECURSE INCLUDES CONFIGURE_DEPENDS ./cgotorch/*.h ./cgotorch/*.hpp)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ./cgotorch/*.c ./cgotorch/*.cc ./cgotorch/*.cpp)

add_library(cgotorch SHARED ${INCLUDES} ${SOURCES})
set_property(TARGET cgotorch PROPERTY CXX_STANDARD 14)
# target_include_directories(cgotorch PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>
#     ./cgotorch
# )
# target_compile_features(cgotorch PUBLIC cxx_std_11)
set_target_properties(cgotorch PROPERTIES
    PUBLIC_HEADER ./cgotorch/cgotorch.hpp
)
target_link_libraries(cgotorch PUBLIC "${TORCH_LIBRARIES}")
