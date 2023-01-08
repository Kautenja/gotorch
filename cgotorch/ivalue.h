// C bindings for c10::IValue.
//
// Copyright (c) 2022 Christian Kauten
// Copyright (c) 2022 Sensory, Inc.
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

// MARK: Constructors

const char* Torch_IValue_FromNone(IValue* output);
// const char* Torch_IValue_FromScalar(IValue* output, IValue data);
const char* Torch_IValue_FromBool(IValue* output, bool data);
const char* Torch_IValue_FromInt(IValue* output, int data);
const char* Torch_IValue_FromDouble(IValue* output, double data);
const char* Torch_IValue_FromComplexDouble(IValue* output, double _Complex data);
const char* Torch_IValue_FromString(IValue* output, const char* data);
const char* Torch_IValue_FromTensor(IValue* output, Tensor data);
const char* Torch_IValue_FromBoolList(IValue* output, bool* data, int num_datums);
const char* Torch_IValue_FromIntList(IValue* output, int* data, int num_datums);
const char* Torch_IValue_FromDoubleList(IValue* output, double* data, int num_datums);
const char* Torch_IValue_FromComplexDoubleList(IValue* output, double _Complex* data, int num_datums);
const char* Torch_IValue_FromTensorList(IValue* output, Tensor* data, int num_datums);
// const char* Torch_IValue_FromOptionalTensorList(IValue* output, IValue data);
// const char* Torch_IValue_FromList(IValue* output, IValue* values, int num_datums);
// const char* Torch_IValue_FromTuple(IValue* output, IValue* values, int num_datums);
const char* Torch_IValue_FromDevice(IValue* output, Device data);
// const char* Torch_IValue_FromStorage(IValue* output, IValue data);
// const char* Torch_IValue_FromCapsule(IValue* output, IValue data);
// const char* Torch_IValue_FromCustomClass(IValue* output, IValue data);
// const char* Torch_IValue_FromFuture(IValue* output, IValue data);
// const char* Torch_IValue_FromRRef(IValue* output, IValue data);
// const char* Torch_IValue_FromQuantizer(IValue* output, IValue data);
// const char* Torch_IValue_FromSymInt(IValue* output, IValue data);
// const char* Torch_IValue_FromSymFloat(IValue* output, IValue data);
// const char* Torch_IValue_FromGenericDict(IValue* output, IValue* keys, IValue* values, int64_t num_datums);
// const char* Torch_IValue_FromObject(IValue* output, IValue data);
// const char* Torch_IValue_FromModule(IValue* output, IValue data);
// const char* Torch_IValue_FromPyObject(IValue* output, IValue data);
// const char* Torch_IValue_FromEnum(IValue* output, IValue data);
// const char* Torch_IValue_FromStream(IValue* output, IValue data);
// const char* Torch_IValue_FromGenerator(IValue* output, IValue data);
// const char* Torch_IValue_FromPtrType(IValue* output, IValue data);

// MARK: Destroyers

void Torch_IValue_Free(IValue value);

// MARK: Type checkers

const char* Torch_IValue_IsNone(bool* output, IValue ivalue);
const char* Torch_IValue_IsScalar(bool* output, IValue ivalue);
const char* Torch_IValue_IsBool(bool* output, IValue ivalue);
const char* Torch_IValue_IsInt(bool* output, IValue ivalue);
const char* Torch_IValue_IsDouble(bool* output, IValue ivalue);
const char* Torch_IValue_IsComplexDouble(bool* output, IValue ivalue);
const char* Torch_IValue_IsString(bool* output, IValue ivalue);
const char* Torch_IValue_IsTensor(bool* output, IValue ivalue);
const char* Torch_IValue_IsTuple(bool* output, IValue ivalue);
const char* Torch_IValue_IsList(bool* output, IValue ivalue);
const char* Torch_IValue_IsBoolList(bool* output, IValue ivalue);
const char* Torch_IValue_IsIntList(bool* output, IValue ivalue);
const char* Torch_IValue_IsDoubleList(bool* output, IValue ivalue);
const char* Torch_IValue_IsComplexDoubleList(bool* output, IValue ivalue);
const char* Torch_IValue_IsTensorList(bool* output, IValue ivalue);
const char* Torch_IValue_IsGenericDict(bool* output, IValue ivalue);
const char* Torch_IValue_IsDevice(bool* output, IValue ivalue);
const char* Torch_IValue_IsStorage(bool* output, IValue ivalue);
const char* Torch_IValue_IsCapsule(bool* output, IValue ivalue);
const char* Torch_IValue_IsCustomClass(bool* output, IValue ivalue);
const char* Torch_IValue_IsFuture(bool* output, IValue ivalue);
const char* Torch_IValue_IsRRef(bool* output, IValue ivalue);
const char* Torch_IValue_IsQuantizer(bool* output, IValue ivalue);
// const char* Torch_IValue_IsSymInt(bool* output, IValue ivalue);
// const char* Torch_IValue_IsSymFloat(bool* output, IValue ivalue);
// const char* Torch_IValue_IsOptionalTensorList(bool* output, IValue ivalue);
const char* Torch_IValue_IsObject(bool* output, IValue ivalue);
const char* Torch_IValue_IsModule(bool* output, IValue ivalue);
const char* Torch_IValue_IsPyObject(bool* output, IValue ivalue);
const char* Torch_IValue_IsEnum(bool* output, IValue ivalue);
const char* Torch_IValue_IsStream(bool* output, IValue ivalue);
const char* Torch_IValue_IsGenerator(bool* output, IValue ivalue);
const char* Torch_IValue_IsPtrType(bool* output, IValue ivalue);

// MARK: Container length checkers

const char* Torch_IValue_LengthTuple(int64_t* output, IValue ivalue);
const char* Torch_IValue_LengthList(int64_t* output, IValue ivalue);
const char* Torch_IValue_LengthDict(int64_t* output, IValue ivalue);

// MARK: Data accessors

const char* Torch_IValue_ToNone(char** output, IValue ivalue);
const char* Torch_IValue_ToBool(bool* output, IValue ivalue);
const char* Torch_IValue_ToInt(int* output, IValue ivalue);
const char* Torch_IValue_ToDouble(double* output, IValue ivalue);
const char* Torch_IValue_ToComplexDouble(double _Complex* output, IValue ivalue);
// const char* Torch_IValue_ToScalar(bool* output, IValue ivalue);
const char* Torch_IValue_ToString(char** output, IValue ivalue);
const char* Torch_IValue_ToTensor(Tensor* output, IValue ivalue);
const char* Torch_IValue_ToBoolList(bool* output, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToIntList(int* output, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToDoubleList(double* output, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToComplexDoubleList(double _Complex* output, int64_t num_datums, IValue ivalue);
// const char* Torch_IValue_ToOptionalTensorList(Tensor* output, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToTensorList(Tensor* output, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToList(IValue* output, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToTuple(IValue* output, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToGenericDict(IValue* keys, IValue* values, int64_t num_datums, IValue ivalue);
const char* Torch_IValue_ToDevice(Device* output, IValue ivalue);
// const char* Torch_IValue_ToStorage(bool* output, IValue ivalue);
// const char* Torch_IValue_ToCapsule(bool* output, IValue ivalue);
// const char* Torch_IValue_ToCustomClass(bool* output, IValue ivalue);
// const char* Torch_IValue_ToFuture(bool* output, IValue ivalue);
// const char* Torch_IValue_ToRRef(bool* output, IValue ivalue);
// const char* Torch_IValue_ToQuantizer(bool* output, IValue ivalue);
// const char* Torch_IValue_ToSymInt(bool* output, IValue ivalue);
// const char* Torch_IValue_ToSymFloat(bool* output, IValue ivalue);
// const char* Torch_IValue_ToObject(bool* output, IValue ivalue);
// const char* Torch_IValue_ToModule(bool* output, IValue ivalue);
// const char* Torch_IValue_ToPyObject(bool* output, IValue ivalue);
// const char* Torch_IValue_ToEnum(bool* output, IValue ivalue);
// const char* Torch_IValue_ToStream(bool* output, IValue ivalue);
// const char* Torch_IValue_ToGenerator(bool* output, IValue ivalue);
// const char* Torch_IValue_ToPtrType(bool* output, IValue ivalue);

#ifdef __cplusplus
}
#endif
