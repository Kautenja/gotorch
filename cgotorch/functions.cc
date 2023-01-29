// C bindings for at::Tensor functions.
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

#include "cgotorch/functions.h"
#include "cgotorch/try_catch_return_error_string.hpp"
#include <vector>
#include <tuple>

// ---------------------------------------------------------------------------
// MARK: Tensor Metadata
// ---------------------------------------------------------------------------

void Torch_Tensor_Numel(int64_t* result, Tensor tensor) {
    *result = tensor->numel();
}

void Torch_Tensor_Is_Complex(bool* result, Tensor tensor) {
    *result = tensor->is_complex();
}

void Torch_Tensor_Is_Conj(bool* result, Tensor tensor) {
    *result = tensor->is_conj();
}

void Torch_Tensor_Is_Floating_Point(bool* result, Tensor tensor) {
    *result = tensor->is_floating_point();
}

void Torch_Tensor_Is_Nonzero(bool* result, Tensor tensor) {
    *result = tensor->is_nonzero();
}

// ---------------------------------------------------------------------------
// MARK: Tensor Creation
// ---------------------------------------------------------------------------

const char* Torch_Zeros(Tensor* result, int64_t* size, int64_t length, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::zeros(torch::IntArrayRef(size, length), *options));
    });
}

const char* Torch_ZerosLike(Tensor* result, Tensor reference) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::zeros_like(*reference));
    });
}

const char* Torch_Ones(Tensor* result, int64_t* size, int64_t length, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::ones(torch::IntArrayRef(size, length), *options));
    });
}

const char* Torch_OnesLike(Tensor* result, Tensor reference) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::ones_like(*reference));
    });
}

const char* Torch_Arange(Tensor* result, float start, float end, float step, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::arange(start, end, step, *options));
    });
}

const char* Torch_Range(Tensor* result, float start, float end, float step, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::range(start, end, step, *options));
    });
}

const char* Torch_Linspace(Tensor* result, float start, float end, int64_t steps, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::linspace(start, end, steps, *options));
    });
}

const char* Torch_Logspace(Tensor* result, float start, float end, int64_t steps, double base, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::logspace(start, end, steps, base, *options));
    });
}

const char* Torch_Eye(Tensor* result, int64_t n, int64_t m, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::eye(n, m, *options));
    });
}

const char* Torch_Empty(Tensor* result, int64_t* size, int64_t length, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::empty(torch::IntArrayRef(size, length), *options));
    });
}

const char* Torch_EmptyLike(Tensor* result, Tensor reference) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::empty_like(*reference));
    });
}

// only for float32
const char* Torch_Full(Tensor* result, int64_t* size, int64_t length, float value, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::full(torch::IntArrayRef(size, length), value, *options));
    });
}

const char* Torch_FullLike(Tensor* result, Tensor reference, float value) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::full_like(*reference, value));
    });
}

const char* Torch_Rand(Tensor* result, int64_t* size, int64_t length, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::rand(torch::IntArrayRef(size, length), *options));
    });
}

const char* Torch_RandLike(Tensor* result, Tensor reference) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::rand_like(*reference));
    });
}

const char* Torch_RandInt(Tensor* result, int64_t* size, int64_t length, int64_t low, int64_t high, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::randint(low, high, torch::IntArrayRef(size, length), *options));
    });
}

const char* Torch_RandIntLike(Tensor* result, Tensor reference, int64_t low, int64_t high) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::randint_like(*reference, low, high));
    });
}

const char* Torch_RandN(Tensor* result, int64_t* size, int64_t length, TensorOptions options) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::randn(torch::IntArrayRef(size, length), *options));
    });
}

const char* Torch_RandNLike(Tensor* result, Tensor reference) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::randn_like(*reference));
    });
}

// MARK: Maths

const char* Torch_Add(Tensor a, Tensor other, float alpha, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::add(*a, *other, alpha));
    });
}

const char* Torch_Add_(Tensor a, Tensor other, float alpha) {
    return try_catch_return_error_string([&] () { a->add_(*other, alpha); });
}

const char* Torch_Sub(Tensor a, Tensor other, float alpha, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::sub(*a, *other, alpha));
    });
}

const char* Torch_Sub_(Tensor a, Tensor other, float alpha) {
    return try_catch_return_error_string([&] () { a->sub_(*other, alpha); });
}

const char* Torch_Mul(Tensor a, Tensor other, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::mul(*a, *other));
    });
}

const char* Torch_Mul_(Tensor a, Tensor other) {
    return try_catch_return_error_string([&] () { a->mul_(*other); });
}

const char* Torch_Div(Tensor a, Tensor other, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::div(*a, *other));
    });
}

const char* Torch_Div_(Tensor a, Tensor other) {
    return try_catch_return_error_string([&] () { a->div_(*other); });
}

const char* Torch_Abs(Tensor a, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->abs());
    });
}

const char* Torch_Abs_(Tensor a) {
    return try_catch_return_error_string([&] () { a->abs_(); });
}

const char* Torch_Square(Tensor a, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->square());
    });
}

const char* Torch_Square_(Tensor a) {
    return try_catch_return_error_string([&] () { a->square_(); });
}

const char* Torch_Sqrt(Tensor a, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->sqrt());
    });
}

const char* Torch_Sqrt_(Tensor a) {
    return try_catch_return_error_string([&] () { a->sqrt_(); });
}

const char* Torch_Pow(Tensor a, double exponent, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(at::pow(*a, exponent));
    });
}

const char* Torch_Tanh(Tensor a, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->tanh());
    });
}

const char* Torch_Sigmoid(Tensor a, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->sigmoid());
    });
}

const char* Torch_LogSoftmax(Tensor a, int64_t dim, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->log_softmax(dim));
    });
}

// MARK: Data layout

const char* Torch_Permute(Tensor a, int64_t* dims, int64_t dims_size, Tensor* result) {
    return try_catch_return_error_string([&] () {
        c10::ArrayRef<int64_t> d(dims, dims_size);
        *result = new at::Tensor(a->permute(d));
    });
}

const char* Torch_Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::transpose(*a, dim0, dim1));
    });
}

const char* Torch_Flatten(Tensor a, int64_t startDim, int64_t endDim, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::flatten(*a, startDim, endDim));
        return nullptr;
    });
}

const char* Torch_Squeeze(Tensor a, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->squeeze());
    });
}

const char* Torch_SqueezeWithDim(Tensor a, int64_t dim, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->squeeze(dim));
    });
}

const char* Torch_Unsqueeze(Tensor* result, Tensor tensor, int64_t dim) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(tensor->unsqueeze(dim));
    });
}

const char* Torch_Stack(Tensor* result, Tensor* tensors, int64_t tensors_size, int64_t dim) {
    return try_catch_return_error_string([&] () {
        std::vector<torch::Tensor> data;
        while (data.size() < tensors_size) data.push_back(**tensors++);
        auto out = at::stack(data, dim);
        *result = new at::Tensor(out);
    });
}

const char* Torch_Cat(Tensor* result, Tensor* tensors, int64_t tensors_size, int64_t dim) {
    return try_catch_return_error_string([&] () {
        std::vector<torch::Tensor> data;
        while (data.size() < tensors_size) data.push_back(**tensors++);
        auto out = at::cat(data, dim);
        *result = new at::Tensor(out);
    });
}

// MARK: Selection

const char* Torch_Slice(Tensor* result, Tensor a, int64_t dim, int64_t start, int64_t end, int64_t step) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::slice(*a, dim, start, end, step));
    });
}

const char* Torch_IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor* result) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(torch::index_select(*a, dim, *index));
    });
}

// ---------------------------------------------------------------------------
// MARK: Reduction Ops
// ---------------------------------------------------------------------------

const char* Torch_Argmin(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->argmin());
    });
}

const char* Torch_ArgminByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->argmin(dim, keep_dims));
    });
}

const char* Torch_Argmax(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->argmax());
    });
}

const char* Torch_ArgmaxByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->argmax(dim, keep_dims));
    });
}

// TODO: amax
// TODO: amin
// TODO: aminmax

const char* Torch_All(Tensor* outputs, Tensor inputs) {
    return try_catch_return_error_string([&] () {
        *outputs = new at::Tensor(torch::all(*inputs));
    });
}

const char* Torch_AllByDim(Tensor* outputs, Tensor inputs, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *outputs = new at::Tensor(torch::all(*inputs, dim, keep_dims));
    });
}

const char* Torch_Any(Tensor* outputs, Tensor inputs) {
    return try_catch_return_error_string([&] () {
        *outputs = new at::Tensor(torch::any(*inputs));
    });
}

const char* Torch_AnyByDim(Tensor* outputs, Tensor inputs, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *outputs = new at::Tensor(torch::any(*inputs, dim, keep_dims));
    });
}

// TODO: dist
// TODO: logsumexp

const char* Torch_Max(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->max());
    });
}

const char* Torch_MaxByDim(Tensor* values, Tensor* indices, Tensor a, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        std::tuple<at::Tensor, at::Tensor> output = a->max(dim, keep_dims);
        *values = new at::Tensor(std::get<0>(output));
        *indices = new at::Tensor(std::get<1>(output));
    });
}

const char* Torch_Min(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->min());
    });
}

const char* Torch_MinByDim(Tensor* values, Tensor* indices, Tensor a, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        std::tuple<at::Tensor, at::Tensor> output = a->min(dim, keep_dims);
        *values = new at::Tensor(std::get<0>(output));
        *indices = new at::Tensor(std::get<1>(output));
    });
}

const char* Torch_Mean(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->mean());
    });
}

const char* Torch_MeanByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->mean(dim, keep_dims));
    });
}

// TODO: nanmean

const char* Torch_Median(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->median());
    });
}

const char* Torch_MedianByDim(Tensor* values, Tensor* indices, Tensor a, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        std::tuple<at::Tensor, at::Tensor> output = a->median(dim, keep_dims);
        *values = new at::Tensor(std::get<0>(output));
        *indices = new at::Tensor(std::get<1>(output));
    });
}

// TODO: nanmedian
// TODO: mode
// TODO: norm
// TODO: nansum
// TODO: prod
// TODO: quantile
// TODO: nanquantile

const char* Torch_Std(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->std());
    });
}

const char* Torch_StdByDim(Tensor* result, Tensor a, int64_t dim, bool unbiased, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->std(dim, unbiased, keep_dims));
    });
}

const char* Torch_StdMean(Tensor* stddev, Tensor* mean, Tensor a) {
    return try_catch_return_error_string([&] () {
        auto output = torch::std_mean(*a);
        *stddev = new at::Tensor(std::get<0>(output));
        *mean = new at::Tensor(std::get<1>(output));
    });
}

const char* Torch_StdMeanByDim(Tensor* stddev, Tensor* mean, Tensor a, int64_t dim, bool unbiased, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        auto output = torch::std_mean(*a, dim, unbiased, keep_dims);
        *stddev = new at::Tensor(std::get<0>(output));
        *mean = new at::Tensor(std::get<1>(output));
    });
}

const char* Torch_Sum(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->sum());
    });
}

const char* Torch_SumByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->sum(dim, keep_dims));
    });
}

// TODO: unique
// TODO: unique_consecutive

const char* Torch_Var(Tensor* result, Tensor a) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->var());
    });
}

const char* Torch_VarByDim(Tensor* result, Tensor a, int64_t dim, bool unbiased, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(a->var(dim, unbiased, keep_dims));
    });
}

const char* Torch_VarMean(Tensor* var, Tensor* mean, Tensor a) {
    return try_catch_return_error_string([&] () {
        auto output = torch::var_mean(*a);
        *var = new at::Tensor(std::get<0>(output));
        *mean = new at::Tensor(std::get<1>(output));
    });
}

const char* Torch_VarMeanByDim(Tensor* var, Tensor* mean, Tensor a, int64_t dim, bool unbiased, bool keep_dims) {
    return try_catch_return_error_string([&] () {
        auto output = torch::var_mean(*a, dim, unbiased, keep_dims);
        *var = new at::Tensor(std::get<0>(output));
        *mean = new at::Tensor(std::get<1>(output));
    });
}

// TODO: count_nonzero

// ---------------------------------------------------------------------------
// MARK: Comparison Ops
// ---------------------------------------------------------------------------

const char* Torch_AllClose(bool* result, Tensor a, Tensor b, double rtol, double atol) {
    return try_catch_return_error_string([&] () {
        *result = at::allclose(*a, *b, rtol, atol);
    });
}

const char* Torch_IsClose(Tensor* result, Tensor a, Tensor b, double rtol, double atol) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(at::isclose(*a, *b, rtol, atol));
    });
}

// TODO: Argsort

const char* Torch_Eq          (Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::eq(*a, *b));            }); }
const char* Torch_Equal       (bool* result,   Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = at::equal(*a, *b);                            }); }
const char* Torch_GreaterEqual(Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::greater_equal(*a, *b)); }); }
const char* Torch_Greater     (Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::greater(*a, *b));       }); }
const char* Torch_LessEqual   (Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::less_equal(*a, *b));    }); }
const char* Torch_Less        (Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::less(*a, *b));          }); }
const char* Torch_IsIn        (Tensor* result, Tensor e, Tensor t) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::isin(*e, *t));          }); }
const char* Torch_Maximum     (Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::maximum(*a, *b));       }); }
const char* Torch_Minimum     (Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::minimum(*a, *b));       }); }
const char* Torch_NotEqual    (Tensor* result, Tensor a, Tensor b) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::not_equal(*a, *b));     }); }

const char* Torch_IsFinite    (Tensor* result, Tensor tensor) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::isfinite(*tensor)); }); }
const char* Torch_IsInf       (Tensor* result, Tensor tensor) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::isinf(*tensor));    }); }
const char* Torch_IsPosInf    (Tensor* result, Tensor tensor) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::isposinf(*tensor)); }); }
const char* Torch_IsNegInf    (Tensor* result, Tensor tensor) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::isneginf(*tensor)); }); }
const char* Torch_IsNan       (Tensor* result, Tensor tensor) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::isnan(*tensor));    }); }
const char* Torch_IsReal      (Tensor* result, Tensor tensor) { return try_catch_return_error_string([&] () { *result = new at::Tensor(torch::isreal(*tensor));   }); }

// TODO: KthValue

const char* Torch_TopK(Tensor* values, Tensor* indices, Tensor a, int64_t k, int64_t dim, int8_t largest, int8_t sorted) {
    return try_catch_return_error_string([&] () {
        auto outputs = torch::topk(*a, k, dim, largest, sorted);
        *values = new at::Tensor(std::get<0>(outputs));
        *indices = new at::Tensor(std::get<1>(outputs));
    });
}

const char* Torch_Sort(Tensor* values, Tensor* indices, Tensor tensor, int64_t dim, bool descending) {
    return try_catch_return_error_string([&] () {
        std::tuple<at::Tensor, at::Tensor> sorted = tensor->sort(dim, descending);
        *values = new at::Tensor(std::get<0>(sorted));
        *indices = new at::Tensor(std::get<1>(sorted));
    });
}

// ---------------------------------------------------------------------------
// MARK: Spectral Operations
// ---------------------------------------------------------------------------

// TODO: stft
// TODO: istft
// TODO: bartlett_window
// TODO: blackman_window
// TODO: hamming_window
// TODO: hann_window
// TODO: kaiser_window

// ---------------------------------------------------------------------------
// MARK: Other Operations
// ---------------------------------------------------------------------------

// TODO: atleast_1d
// TODO: atleast_2d
// TODO: atleast_3d
// TODO: bincount
// TODO: block_diag
// TODO: broadcast_tensors
// TODO: broadcast_to
// TODO: broadcast_shapes
// TODO: bucketize
// TODO: cartesian_prod
// TODO: cdist
// TODO: clone
// TODO: combinations
// TODO: corrcoef
// TODO: cov
// TODO: cross
// TODO: cummax
// TODO: cummin
// TODO: cumprod
// TODO: cumsum
// TODO: diag
// TODO: diag_embed
// TODO: diagflat
// TODO: diagonal
// TODO: diff
// TODO: einsum
// TODO: flatten
// TODO: flip
// TODO: fliplr
// TODO: flipud
// TODO: kron
// TODO: rot90
// TODO: gcd
// TODO: histc
// TODO: histogram
// TODO: histogramdd
// TODO: meshgrid
// TODO: lcm
// TODO: logcumsumexp
// TODO: ravel
// TODO: renorm
// TODO: repeat_interleave
// TODO: roll
// TODO: searchsorted
// TODO: tensordot
// TODO: trace
// TODO: tril
// TODO: tril_indices
// TODO: triu
// TODO: triu_indices
// TODO: unflatten
// TODO: vander
// TODO: view_as_real
// TODO: view_as_complex
// TODO: resolve_conj
// TODO: resolve_neg

// ---------------------------------------------------------------------------
// MARK: BLAS and LAPACK Operations
// ---------------------------------------------------------------------------

// TODO: addbmm
// TODO: addmm
// TODO: addmv
// TODO: addr
// TODO: baddbmm
// TODO: bmm
// TODO: chain_matmul
// TODO: cholesky
// TODO: cholesky_inverse
// TODO: cholesky_solve
// TODO: dot
// TODO: geqrf
// TODO: ger
// TODO: inner
// TODO: inverse
// TODO: det
// TODO: logdet
// TODO: slogdet
// TODO: lu
// TODO: lu_solve
// TODO: lu_unpack
// TODO: matmul
// TODO: matrix_power
// TODO: matrix_exp

const char* Torch_MM(Tensor* result, Tensor a, Tensor b) {
    return try_catch_return_error_string([&] () {
        *result = new at::Tensor(at::mm(*a, *b));
    });
}

// TODO: mv
// TODO: orgqr
// TODO: ormqr
// TODO: outer
// TODO: pinverse
// TODO: qr
// TODO: svd
// TODO: svd_lowrank
// TODO: pca_lowrank
// TODO: symeig
// TODO: lobpcg
// TODO: trapz
// TODO: trapezoid
// TODO: cumulative_trapezoid
// TODO: triangular_solve
// TODO: vdot
