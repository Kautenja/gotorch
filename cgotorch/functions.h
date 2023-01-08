// C bindings for at::Tensor functions.
//
// Copyright (c) 2022 Christian Kauten
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

// ---------------------------------------------------------------------------
// MARK: Tensor Metadata
// ---------------------------------------------------------------------------

void Torch_Tensor_Numel(int64_t* result, Tensor tensor);
void Torch_Tensor_Is_Complex(bool* result, Tensor tensor);
void Torch_Tensor_Is_Conj(bool* result, Tensor tensor);
void Torch_Tensor_Is_Floating_Point(bool* result, Tensor tensor);
void Torch_Tensor_Is_Nonzero(bool* result, Tensor tensor);

// ---------------------------------------------------------------------------
// MARK: Tensor Creation
// ---------------------------------------------------------------------------

const char* Torch_Zeros(Tensor* result, int64_t* size, int64_t length, TensorOptions options);
const char* Torch_ZerosLike(Tensor* result, Tensor reference);
const char* Torch_Ones(Tensor* result, int64_t* size, int64_t length, TensorOptions options);
const char* Torch_OnesLike(Tensor* result, Tensor reference);
const char* Torch_Arange(Tensor* result, float start, float end, float step, TensorOptions options);
const char* Torch_Range(Tensor* result, float start, float end, float step, TensorOptions options);
const char* Torch_Linspace(Tensor* result, float start, float end, int64_t steps, TensorOptions options);
const char* Torch_Logspace(Tensor* result, float start, float end, int64_t steps, double base, TensorOptions options);
const char* Torch_Eye(Tensor* result, int64_t n, int64_t m, TensorOptions options);
const char* Torch_Empty(Tensor* result, int64_t* size, int64_t length, TensorOptions options);
const char* Torch_EmptyLike(Tensor* result, Tensor reference);
const char* Torch_Full(Tensor* result, int64_t* size, int64_t length, float value, TensorOptions options);
const char* Torch_FullLike(Tensor* result, Tensor reference, float value);
// const char* Torch_Bernoulli(Tensor* result, Tensor reference);
// const char* Torch_Multinomial(Tensor* result, Tensor reference);
// const char* Torch_Normal(Tensor* result, Tensor reference);
// const char* Torch_Poisson(Tensor* result, Tensor reference);
const char* Torch_Rand(Tensor* result, int64_t* size, int64_t length, TensorOptions options);
const char* Torch_RandLike(Tensor* result, Tensor reference);
const char* Torch_RandInt(Tensor* result, int64_t* size, int64_t length, int64_t low, int64_t high, TensorOptions options);
const char* Torch_RandIntLike(Tensor* result, Tensor reference, int64_t low, int64_t high);
const char* Torch_RandN(Tensor* result, int64_t* size, int64_t length, TensorOptions options);
const char* Torch_RandNLike(Tensor* result, Tensor reference);
// const char* Torch_RandPerm();

// ---------------------------------------------------------------------------
// MARK: Indexing, Slicing, Joining, Mutating Ops
// ---------------------------------------------------------------------------

// TODO: adjoint
// TODO: argwhere
const char* Torch_Cat(Tensor* result, Tensor* tensors, int64_t tensors_size, int64_t dim);
const char* Torch_Stack(Tensor* result, Tensor* tensors, int64_t tensors_size, int64_t dim);
// TODO: Conj
// TODO: chunk
// TODO: dsplit
// TODO: column_stack
// TODO: dstack
// TODO: gather
// TODO: hsplit
// TODO: hstack
// TODO: index_add
// TODO: index_copy
// TODO: index_reduce
const char* Torch_IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor* result);
// TODO: masked_select
// TODO: movedim
// TODO: moveaxis
// TODO: narrow
// TODO: nonzero
const char* Torch_Permute(Tensor a, int64_t* dims, int64_t dims_size, Tensor* result);
// TODO: row_stack
// TODO: select
// TODO: scatter
// TODO: diagonal_scatter
// TODO: select_scatter
const char* Torch_Slice(Tensor* result, Tensor a, int64_t dim, int64_t start, int64_t end, int64_t step);
// TODO: slice_scatter
// TODO: scatter_add
// TODO: scatter_reduce
// TODO: split
const char* Torch_Squeeze(Tensor a, Tensor* result);
const char* Torch_SqueezeWithDim(Tensor a, int64_t dim, Tensor* result);
// TODO: stack
// TODO: swapaxes
// TODO: swapdims
// TODO: t
// TODO: take
// TODO: take_along_dim
// TODO: tensor_split
// TODO: tile
const char* Torch_Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor* result);
// TODO: unbind
const char* Torch_Unsqueeze(Tensor* result, Tensor tensor, int64_t dim);
// TODO: vsplit
// TODO: vstack
// TODO: where

// ---------------------------------------------------------------------------
// MARK: Pointwise Ops
// ---------------------------------------------------------------------------

const char* Torch_Abs(Tensor a, Tensor* result);
const char* Torch_Abs_(Tensor a);
// TODO: absolute
// TODO: acos
// TODO: arccos
// TODO: acosh
// TODO: arccosh
const char* Torch_Add(Tensor a, Tensor other, float alpha, Tensor* result);
const char* Torch_Add_(Tensor a, Tensor other, float alpha);
// TODO: addcdiv
// TODO: addcmul
// TODO: angle
// TODO: asin
// TODO: arcsin
// TODO: asinh
// TODO: arcsinh
// TODO: atan
// TODO: arctan
// TODO: atanh
// TODO: arctanh
// TODO: atan2
// TODO: arctan2
// TODO: bitwise_not
// TODO: bitwise_and
// TODO: bitwise_or
// TODO: bitwise_xor
// TODO: bitwise_left_shift
// TODO: bitwise_right_shift
// TODO: ceil
// TODO: clamp
// TODO: clip
// TODO: conj_physical
// TODO: copysign
// TODO: cos
// TODO: cosh
// TODO: deg2rad
const char* Torch_Div(Tensor a, Tensor other, Tensor* result);
const char* Torch_Div_(Tensor a, Tensor other);
// TODO: divide
// TODO: digamma
// TODO: erf
// TODO: erfc
// TODO: erfinv
// TODO: exp
// TODO: exp2
// TODO: expm1
// TODO: fake_quantize_per_channel_affine
// TODO: fake_quantize_per_tensor_affine
// TODO: fix
// TODO: float_power
// TODO: floor
// TODO: floor_divide
// TODO: fmod
// TODO: frac
// TODO: frexp
// TODO: gradient
// TODO: imag
// TODO: ldexp
// TODO: lerp
// TODO: lgamma
// TODO: log
const char* Torch_LogSoftmax(Tensor a, int64_t dim, Tensor* result);
// TODO: log10
// TODO: log1p
// TODO: log2
// TODO: logaddexp
// TODO: logaddexp2
// TODO: logical_and
// TODO: logical_not
// TODO: logical_or
// TODO: logical_xor
// TODO: logit
// TODO: hypot
// TODO: i0
// TODO: igamma
// TODO: igammac
const char* Torch_Mul(Tensor a, Tensor other, Tensor* result);
const char* Torch_Mul_(Tensor a, Tensor other);
// TODO: multiply
// TODO: mvlgamma
// TODO: nan_to_num
// TODO: neg
// TODO: negative
// TODO: nextafter
// TODO: polygamma
// TODO: positive
const char* Torch_Pow(Tensor a, double exponent, Tensor* result);
// TODO: quantized_batch_norm
// TODO: quantized_max_pool1d
// TODO: quantized_max_pool2d
// TODO: rad2deg
// TODO: real
// TODO: reciprocal
// TODO: remainder
// TODO: round
// TODO: rsqrt
const char* Torch_Sigmoid(Tensor a, Tensor* result);
// TODO: sign
// TODO: sgn
// TODO: signbit
// TODO: sin
// TODO: sinc
// TODO: sinh
const char* Torch_Sqrt(Tensor a, Tensor* result);
const char* Torch_Sqrt_(Tensor a);
const char* Torch_Square(Tensor a, Tensor* result);
const char* Torch_Square_(Tensor a);
const char* Torch_Sub(Tensor a, Tensor other, float alpha, Tensor* result);
const char* Torch_Sub_(Tensor a, Tensor other, float alpha);
// TODO: subtract
// TODO: tan
const char* Torch_Tanh(Tensor a, Tensor* result);
// TODO: true_divide
// TODO: trunc

// ---------------------------------------------------------------------------
// MARK: Reduction Ops
// ---------------------------------------------------------------------------

const char* Torch_Argmin(Tensor* result, Tensor a);
const char* Torch_ArgminByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims);
const char* Torch_Argmax(Tensor* result, Tensor a);
const char* Torch_ArgmaxByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims);
// TODO: amax
// TODO: amin
// TODO: aminmax
const char* Torch_All(Tensor* result, Tensor inputs);
const char* Torch_AllByDim(Tensor* result, Tensor inputs, int64_t dim, bool keep_dims);
const char* Torch_Any(Tensor* result, Tensor inputs);
const char* Torch_AnyByDim(Tensor* result, Tensor inputs, int64_t dim, bool keep_dims);
// TODO: dist
// TODO: logsumexp
const char* Torch_Max(Tensor* result, Tensor a);
const char* Torch_MaxByDim(Tensor* values, Tensor* indices, Tensor a, int64_t dim, bool keep_dims);
const char* Torch_Min(Tensor* result, Tensor a);
const char* Torch_MinByDim(Tensor* values, Tensor* indices, Tensor a, int64_t dim, bool keep_dims);
const char* Torch_Mean(Tensor* result, Tensor a);
const char* Torch_MeanByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims);
// TODO: nanmean
const char* Torch_Median(Tensor* result, Tensor a);
const char* Torch_MedianByDim(Tensor* values, Tensor* indices, Tensor a, int64_t dim, bool keep_dims);
// TODO: nanmedian
// TODO: mode
// TODO: norm
// TODO: nansum
// TODO: prod
// TODO: quantile
// TODO: nanquantile
const char* Torch_Std(Tensor* result, Tensor a);
const char* Torch_StdByDim(Tensor* result, Tensor a, int64_t dim, bool unbiased, bool keep_dims);
const char* Torch_StdMean(Tensor* stddev, Tensor* mean, Tensor a);
const char* Torch_StdMeanByDim(Tensor* stddev, Tensor* mean, Tensor a, int64_t dim, bool unbiased, bool keep_dims);
const char* Torch_Sum(Tensor* result, Tensor a);
const char* Torch_SumByDim(Tensor* result, Tensor a, int64_t dim, bool keep_dims);
// TODO: Torch_Unique
// TODO: Torch_UniqueConsecutive
const char* Torch_Var(Tensor* result, Tensor a);
const char* Torch_VarByDim(Tensor* result, Tensor a, int64_t dim, bool unbiased, bool keep_dims);
const char* Torch_VarMean(Tensor* var, Tensor* mean, Tensor a);
const char* Torch_VarMeanByDim(Tensor* var, Tensor* mean, Tensor a, int64_t dim, bool unbiased, bool keep_dims);
// TODO: Torch_CountNonzero

// ---------------------------------------------------------------------------
// MARK: Comparison Ops
// ---------------------------------------------------------------------------

const char* Torch_AllClose(bool* result, Tensor a, Tensor b, double rtol, double atol);
const char* Torch_IsClose(Tensor* result, Tensor a, Tensor b, double rtol, double atol);
// TODO: Torch_Argsort
const char* Torch_Eq(Tensor* result, Tensor a, Tensor b);
const char* Torch_Equal(bool* result, Tensor a, Tensor b);
const char* Torch_GreaterEqual(Tensor* result, Tensor a, Tensor b);
const char* Torch_Greater(Tensor* result, Tensor a, Tensor b);
const char* Torch_LessEqual(Tensor* result, Tensor a, Tensor b);
const char* Torch_Less(Tensor* result, Tensor a, Tensor b);
const char* Torch_IsIn(Tensor* result, Tensor elements, Tensor test_elements);
const char* Torch_Maximum(Tensor* result, Tensor a, Tensor b);
const char* Torch_Minimum(Tensor* result, Tensor a, Tensor b);
const char* Torch_NotEqual(Tensor* result, Tensor a, Tensor b);

const char* Torch_IsFinite(Tensor* result, Tensor tensor);
const char* Torch_IsInf(Tensor* result, Tensor tensor);
const char* Torch_IsPosInf(Tensor* result, Tensor tensor);
const char* Torch_IsNegInf(Tensor* result, Tensor tensor);
const char* Torch_IsNan(Tensor* result, Tensor tensor);
const char* Torch_IsReal(Tensor* result, Tensor tensor);
// TODO: Torch_KthValue
const char* Torch_TopK(Tensor* values, Tensor* indices, Tensor a, int64_t k, int64_t dim, int8_t largest, int8_t sorted);
const char* Torch_Sort(Tensor* values, Tensor* indices, Tensor tensor, int64_t dim, bool descending);

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
const char* Torch_Flatten(Tensor a, int64_t startDim, int64_t endDim, Tensor* result);
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

const char* Torch_MM(Tensor* result, Tensor a, Tensor b);

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

#ifdef __cplusplus
}
#endif
