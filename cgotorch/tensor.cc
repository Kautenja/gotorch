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

#include "cgotorch/tensor.h"
#include "cgotorch/try_catch_return_error_string.hpp"
#include <string>
#include <vector>

const char* Torch_FromBlob(Tensor* output,
    void* data,
    int8_t dtype,
    int64_t* size,
    int64_t num_dims
) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(at::from_blob(data,
            at::IntArrayRef(size, num_dims),
            torch::dtype(at::ScalarType(dtype))));
    });
}

const char* Torch_Tensor(Tensor* output,
    void* data,
    int8_t dtype,
    int64_t* size,
    int64_t num_dims
) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(at::from_blob(data,
            at::IntArrayRef(size, num_dims),
            torch::dtype(at::ScalarType(dtype))).clone());
    });
}

void Torch_Tensor_Close(Tensor tensor) { delete tensor; }

const char* Torch_Tensor_Clone(Tensor* output, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(tensor->clone());
    });
}

const char* Torch_Tensor_String(Tensor tensor) {
    std::stringstream ss;
    ss << *tensor;
    std::string s = ss.str();
    char* output = reinterpret_cast<char*>(malloc(s.size() + 1));
    snprintf(output, s.size() + 1, "%s", s.c_str());
    return output;
}

const char* Torch_Tensor_Save(const char* path, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        torch::save(*tensor, std::string(path));
    });
}

const char* Torch_Tensor_Load(Tensor* tensor, const char* path) {
    return try_catch_return_error_string([&] () {
        *tensor = new at::Tensor();
        torch::load(**tensor, path);
    });
}

const char* Torch_Tensor_Encode(ByteBuffer* output, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        *output = new (std::vector<char>);
        **output = torch::pickle_save(tensor->cpu());
    });
}

const char* Torch_Tensor_Decode(Tensor* r, void* addr, int64_t size) {
    return try_catch_return_error_string([&] () {
        auto data = static_cast<const char*>(addr);
        std::vector<char> buf(data, data + static_cast<int>(size));
        *r = new at::Tensor();
        **r = torch::pickle_load(buf).toTensor();
    });
}

const char* Torch_Tensor_Dtype(int8_t* dtype, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        auto t = tensor->scalar_type();
        *dtype = static_cast<int8_t>(t);
    });
}

const char* Torch_Tensor_Dim(int64_t* dim, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        *dim = tensor->dim();
    });
}

const char* Torch_Tensor_Shape(int64_t* dims, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        int i = 0;
        for (int64_t dim : tensor->sizes()) dims[i++] = dim;
    });
}

const char* Torch_Tensor_View(Tensor* output, Tensor tensor, int64_t *size, int64_t size_len) {
  return try_catch_return_error_string([&] () {
    *output = new at::Tensor(tensor->view(torch::IntArrayRef(size, size_len)));
  });
}

const char* Torch_Tensor_ViewAs(Tensor* output, Tensor tensor, Tensor other) {
  return try_catch_return_error_string([&] () {
    *output = new at::Tensor(tensor->view_as(*other));
  });
}

const char* Torch_Tensor_Reshape(Tensor* output, Tensor tensor, int64_t *size, int64_t size_len) {
  return try_catch_return_error_string([&] () {
    *output = new at::Tensor(tensor->reshape(torch::IntArrayRef(size, size_len)));
  });
}

const char* Torch_Tensor_ReshapeAs(Tensor* output, Tensor tensor, Tensor other) {
  return try_catch_return_error_string([&] () {
    *output = new at::Tensor(tensor->reshape_as(*other));
  });
}

const char* Torch_Tensor_Expand(Tensor* output, Tensor tensor, int64_t *size, int64_t size_len) {
  return try_catch_return_error_string([&] () {
    *output = new at::Tensor(tensor->expand(torch::IntArrayRef(size, size_len)));
  });
}

const char* Torch_Tensor_ExpandAs(Tensor* output, Tensor tensor, Tensor other) {
  return try_catch_return_error_string([&] () {
    *output = new at::Tensor(tensor->expand_as(*other));
  });
}

const char* Torch_Tensor_SetData(Tensor tensor, Tensor other) {
    return try_catch_return_error_string([&] () {
        tensor->set_data(*other);
    });
}

const char* Torch_Tensor_Copy_(Tensor tensor, Tensor other) {
    return try_catch_return_error_string([&] () {
        tensor->copy_(*other);
    });
}

const char* Torch_Tensor_CastTo(Tensor* output, Tensor tensor, int8_t dtype) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(tensor->to(static_cast<at::ScalarType>(dtype)));
    });
}

const char* Torch_Tensor_CopyTo(Tensor* output, Tensor tensor, Device device) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(tensor->to(*device));
    });
}

const char* Torch_Tensor_To(Tensor* output, Tensor tensor, Device device, int8_t dtype) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(tensor->to(*device, static_cast<at::ScalarType>(dtype)));
    });
}

const char* Torch_Tensor_PinMemory(Tensor* output, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(tensor->pin_memory());
    });
}

void Torch_Tensor_RequiresGrad(bool* requires_grad, Tensor tensor) {
    *requires_grad = tensor->requires_grad();
}

void Torch_Tensor_SetRequiresGrad(Tensor tensor, bool requires_grad) {
    tensor->set_requires_grad(requires_grad);
}

const char* Torch_Tensor_Backward(Tensor tensor) {
    return try_catch_return_error_string([&] () { tensor->backward(); });
}

const char* Torch_Tensor_Grad(Tensor* output, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(tensor->grad());
    });
}

const char* Torch_Tensor_Detach(Tensor* output, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        *output = new at::Tensor(tensor->detach());
    });
}

// const char* Torch_Tensor_Index(Tensor* output, Tensor input, int64_t* index, int64_t index_len) {
//     return try_catch_return_error_string([&] () {
//         std::vector<at::indexing::TensorIndex> indices;
//         for (int i = 0; i < static_cast<int>(index_len); i++)
//             indices.push_back(at::indexing::TensorIndex(index[i]));
//         at::ArrayRef<at::indexing::TensorIndex> ref(indices.data(), index_len);
//         *output = new at::Tensor(input->index(ref));
//     });
// }

const char* Torch_Tensor_Index(Tensor* output, Tensor tensor, Tensor index) { return try_catch_return_error_string([&] () { *output = new at::Tensor(tensor->index({*index})); }); }

const char* Torch_Tensor_ItemUint8  (uint8_t* output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<uint8_t>(); }); }
const char* Torch_Tensor_ItemInt8   (int8_t*  output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<int8_t >(); }); }
const char* Torch_Tensor_ItemInt16  (int16_t* output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<int16_t>(); }); }
const char* Torch_Tensor_ItemInt32  (int32_t* output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<int32_t>(); }); }
const char* Torch_Tensor_ItemInt64  (int64_t* output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<int64_t>(); }); }
const char* Torch_Tensor_ItemFloat32(float*   output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<float  >(); }); }
const char* Torch_Tensor_ItemFloat64(double*  output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<double >(); }); }
// TODO: Torch_Tensor_ItemComplexHalf
// TODO: Torch_Tensor_ItemComplexFloat
// TODO: Torch_Tensor_ItemComplexDouble
const char* Torch_Tensor_ItemBool   (bool*    output, Tensor a) { return try_catch_return_error_string([&] () { *output = a->item<bool   >(); }); }
// TODO: Torch_Tensor_ItemQInt8
// TODO: Torch_Tensor_ItemQUInt8
// TODO: Torch_Tensor_ItemQInt32
// TODO: Torch_Tensor_ItemBFloat16

const char* Torch_Tensor_ToBytes(uint8_t** buffer, Tensor tensor) {
    return try_catch_return_error_string([&] () {
        *buffer = reinterpret_cast<uint8_t*>(tensor->data_ptr());
    });
}
