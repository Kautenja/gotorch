// C bindings for at::Tensor.
//
// Copyright (c) 2023 Christian Kauten
// Copyright (c) 2022 Sensory, Inc.
// Copyright (c) 2020 GoTorch Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXTERNRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Create a tensor pointer to an existing memory buffer.
/// @param output The output to store the new Tensor pointer in.
/// @param data The pointer to the data buffer.
/// @param dtype The data-type of the data in the buffer
/// @param size The size of the data buffer (as a list of integers.)
/// @param num_dims The cardinality of the `sizes_data` buffer.
/// @returns A dynamically allocated message if an error occurs, else a nullptr.
/// @details
/// This function does not allocate new space, in-place operation performed on
/// the tensor **will** mutate the input data.
const char* Torch_FromBlob(Tensor* output, void* data, int8_t dtype, int64_t* size, int64_t num_dims);

/// @brief Create a tensor pointer to an existing memory buffer.
/// @param result The output to store the new Tensor pointer in.
/// @param data The pointer to the data buffer.
/// @param dtype The data-type of the data in the buffer
/// @param size The size of the data buffer (as a list of integers.)
/// @param num_dims The cardinality of the `sizes_data` buffer.
/// @returns A dynamically allocated message if an error occurs, else a nullptr.
/// @details
/// This function allocates new space, in-place operation performed on the
/// tensor **will not** mutate the input data.
const char* Torch_Tensor(Tensor* output, void* data, int8_t dtype, int64_t* size, int64_t num_dims);

/// @brief De-allocate the given heap allocated tensor.
/// @param tensor An existing tensor that has been allocated by CGoTorch.
void Torch_Tensor_Close(Tensor tensor);

/// @brief Create a new tensor by deep copy of an existing tensor.
/// @param tensor The tensor to create a deep copy of
/// @param output A pointer to store the resulting tensor pointer in.
/// @returns A dynamically allocated message if an error occurs, else a nullptr.
const char* Torch_Tensor_Clone(Tensor* output, Tensor tensor);

/// @brief Convert the tensor to a string representation.
/// @param a The tensor to convert into a string
/// @returns A character pointer to a string representation.
/// @details
/// Ownership of the output character buffer is transferred to the caller. The
/// memory should be released using `std::free` when done. **DO NOT** use the
/// `delete` operator to free the memory, it is allocated in C using `malloc`.
const char* Torch_Tensor_String(Tensor tensor);

/// @brief Serialize a tensor to a binary format on the filesystem.
/// @param path The path to save the tensor to on the filesystem.
/// @param tensor The tensor to save.
const char* Torch_Tensor_Save(const char* path, Tensor tensor);

/// @brief De-serialize a tensor from a binary format on the filesystem.
/// @param output A pointer for the allocated tensor data.
/// @param path The path to load the tensor from on the filesystem.
const char* Torch_Tensor_Load(Tensor* output, const char* path);

/// @brief Encode a tensor to a pickled representation.
/// @param output The output buffer to populate with encoded tensor data.
/// @param tensor The tensor to encode.
const char* Torch_Tensor_Encode(ByteBuffer* output, Tensor tensor);

/// @brief Decode a binary tensor into a structured representation.
/// @param output A pointer to a new tensor to create
/// @param addr A pointer to the compressed tensor.
/// @param size The number of bytes in the binary buffer.
const char* Torch_Tensor_Decode(Tensor* output, void* addr, int64_t size);

const char* Torch_Tensor_Dtype(int8_t *dtype, Tensor tensor);
const char* Torch_Tensor_Dim(int64_t *dim, Tensor tensor);
const char* Torch_Tensor_Shape(int64_t *dims, Tensor tensor);
const char* Torch_Tensor_View(Tensor* output, Tensor tensor, int64_t *size, int64_t size_len);
const char* Torch_Tensor_ViewAs(Tensor* output, Tensor tensor, Tensor other);
const char* Torch_Tensor_Reshape(Tensor* output, Tensor tensor, int64_t *size, int64_t size_len);
const char* Torch_Tensor_ReshapeAs(Tensor* output, Tensor tensor, Tensor other);
const char* Torch_Tensor_Expand(Tensor* output, Tensor tensor, int64_t *size, int64_t size_len);
const char* Torch_Tensor_ExpandAs(Tensor* output, Tensor tensor, Tensor other);
const char* Torch_Tensor_SetData(Tensor tensor, Tensor other);
const char* Torch_Tensor_Copy_(Tensor tensor, Tensor other);
const char* Torch_Tensor_CastTo(Tensor* output, Tensor tensor, int8_t dtype);
const char* Torch_Tensor_CopyTo(Tensor* output, Tensor tensor, Device device);
const char* Torch_Tensor_To(Tensor* output, Tensor tensor, Device device, int8_t dtype);
const char* Torch_Tensor_PinMemory(Tensor* output, Tensor tensor);

/// @brief Access the `requires_grad` property of a tensor.
/// @param requires_grad A pointer to a return register for the boolean value.
/// @param tensor The tensor to access the `requires_grad` property of.
void Torch_Tensor_RequiresGrad(bool* requires_grad, Tensor tensor);

/// @brief Set the `requires_grad` property of a tensor.
/// @param tensor The tensor to set the `requires_grad` property of.
/// @param requires_grad True to enable gradient generation, false otherwise.
void Torch_Tensor_SetRequiresGrad(Tensor tensor, bool requires_grad);

/// @brief Compute the backward pass with taped gradients.
/// @param tensor The tensor to backward pass through.
const char* Torch_Tensor_Backward(Tensor tensor);

/// @brief Get the gradients of the given tensor.
/// @param output A pointer to the buffer to store the result in.
/// @param tensor The tensor to get the gradients of.
const char* Torch_Tensor_Grad(Tensor* output, Tensor tensor);

/// @param tensor The tensor to create a deep copy of
/// @param output A pointer to store the detached tensor's pointer in.
/// @returns A dynamically allocated message if an error occurs, else a nullptr.
const char* Torch_Tensor_Detach(Tensor* output, Tensor tensor);

/// @brief Convert an N-dimensional tensor to a scalar by index.
/// @param output A pointer to the buffer to store the result in.
/// @param tensor The tensor to select a scalar quantity from.
/// @param index A pointer to the coordinates of the value to select.
/// @param index_len The length of the input coordinate array.
/// @returns A dynamically allocated message if an error occurs, else a nullptr.
// const char* Torch_Tensor_Index(Tensor* output, Tensor tensor, int64_t *index, int64_t index_len);

const char* Torch_Tensor_Index(Tensor* output, Tensor tensor, Tensor index);

// @brief Convert a 0-dimensional tensor to a scalar.
// @param output A pointer to the buffer to store the result in.
// @param tensor The tensor to convert to a scalar quantity.
// @returns A dynamically allocated message if an error occurs, else a nullptr.

const char* Torch_Tensor_ItemUint8(uint8_t* output, Tensor tensor);
const char* Torch_Tensor_ItemInt8(int8_t* output, Tensor tensor);
const char* Torch_Tensor_ItemInt16(int16_t* output, Tensor tensor);
const char* Torch_Tensor_ItemInt32(int32_t* output, Tensor tensor);
const char* Torch_Tensor_ItemInt64(int64_t* output, Tensor tensor);
// const char* Torch_Tensor_ItemFloat16(float* output, Tensor tensor);
const char* Torch_Tensor_ItemFloat32(float* output, Tensor tensor);
const char* Torch_Tensor_ItemFloat64(double* output, Tensor tensor);
// const char* Torch_Tensor_ItemComplexHalf(TODO, Tensor tensor);
// const char* Torch_Tensor_ItemComplexFloat(TODO, Tensor tensor);
// const char* Torch_Tensor_ItemComplexDouble(TODO, Tensor tensor);
const char* Torch_Tensor_ItemBool(bool* output, Tensor tensor);
// const char* Torch_Tensor_ItemQInt8(TODO, Tensor tensor);
// const char* Torch_Tensor_ItemQUInt8(TODO, Tensor tensor);
// const char* Torch_Tensor_ItemQInt32(TODO, Tensor tensor);
// const char* Torch_Tensor_ItemBFloat16(TODO, Tensor tensor);

/// @brief Access the contents of a tensor as a raw binary buffer.
/// @param buffer The pointer to the output buffer to update to pointer to
/// raw tensor data.
/// @param tensor The tensor to access the raw contents of.
/// @returns A dynamically allocated message if an error occurs, else a nullptr.
/// @details
/// This function does not allocate new space, in-place operation performed on
/// the buffer **will** mutate the input data.
const char* Torch_Tensor_ToBytes(uint8_t** buffer, Tensor tensor);

#ifdef __cplusplus
}
#endif
