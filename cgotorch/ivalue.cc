// C bindings for c10::IValue.
//
// Copyright (c) 2023 Christian Kauten
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

#include <complex>
#include <vector>
#include "cgotorch/try_catch_return_error_string.hpp"
#include "cgotorch/ivalue.h"

// MARK: Constructors

const char* Torch_IValue_FromNone(IValue* output) { return try_catch_return_error_string([&]() { *output = new torch::IValue(); }); }
// const char* Torch_IValue_FromScalar(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
const char* Torch_IValue_FromBool         (IValue* output, bool            data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(data);  }); }
const char* Torch_IValue_FromInt          (IValue* output, int             data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(data);  }); }
const char* Torch_IValue_FromDouble       (IValue* output, double          data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(data);  }); }
const char* Torch_IValue_FromComplexDouble(IValue* output, double _Complex data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(reinterpret_cast<c10::complex<double>(&)>(data)); }); }
const char* Torch_IValue_FromString       (IValue* output, const char* data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(data);  }); }
const char* Torch_IValue_FromTensor       (IValue* output, Tensor      data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
const char* Torch_IValue_FromBoolList         (IValue* output, bool*            data, int num_datums) { return try_catch_return_error_string([&]() { *output = new torch::IValue(std::vector<bool>(data, data + num_datums));            }); }
const char* Torch_IValue_FromIntList          (IValue* output, int*             data, int num_datums) { return try_catch_return_error_string([&]() { *output = new torch::IValue(std::vector<int>(data, data + num_datums));             }); }
const char* Torch_IValue_FromDoubleList       (IValue* output, double*          data, int num_datums) { return try_catch_return_error_string([&]() { *output = new torch::IValue(std::vector<double>(data, data + num_datums));          }); }
const char* Torch_IValue_FromComplexDoubleList(IValue* output, double _Complex* data, int num_datums) { return try_catch_return_error_string([&]() { *output = new torch::IValue(std::vector<double _Complex>(data, data + num_datums)); }); }
const char* Torch_IValue_FromTensorList       (IValue* output, Tensor*          data, int num_datums) {
    return try_catch_return_error_string([&]() {
        std::vector<torch::Tensor> inputs;
        for (int i = 0; i < num_datums; i++) inputs.push_back(*data[i]);
        *output = new torch::IValue(inputs);
    });
}
// const char* Torch_IValue_FromOptionalTensorList(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromList(IValue* output, IValue* data, int num_datums) {
//     return try_catch_return_error_string([&]() {
//         std::vector<torch::IValue> inputs;
//         for (int i = 0; i < num_datums; i++) inputs.push_back(*data[i]);
//         *output = new torch::IValue(inputs);
//     });
// }
// const char* Torch_IValue_FromTuple(IValue* output, IValue* data, int num_datums) { return Torch_IValue_FromList(output, data, num_datums); }
const char* Torch_IValue_FromDevice(IValue* output, Device data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromStorage(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromCapsule(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromCustomClass(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromFuture(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromRRef(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromQuantizer(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromSymInt(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromGenericDict(IValue* output, IValue* keys, IValue* values, int64_t num_datums) { }
// const char* Torch_IValue_FromObject(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromModule(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromPyObject(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromEnum(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromStream(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromGenerator(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }
// const char* Torch_IValue_FromPtrType(IValue* output, IValue data) { return try_catch_return_error_string([&]() { *output = new torch::IValue(*data); }); }

// MARK: Destroyers

void Torch_IValue_Free(IValue value) { delete value; }

// MARK: Type checkers

const char* Torch_IValue_IsNone              (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isNone();               }); }
const char* Torch_IValue_IsScalar            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isScalar();             }); }
const char* Torch_IValue_IsBool              (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isBool();               }); }
const char* Torch_IValue_IsInt               (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isInt();                }); }
const char* Torch_IValue_IsDouble            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isDouble();             }); }
const char* Torch_IValue_IsComplexDouble     (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isComplexDouble();      }); }
const char* Torch_IValue_IsString            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isString();             }); }
const char* Torch_IValue_IsTensor            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isTensor();             }); }
const char* Torch_IValue_IsTuple             (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isTuple();              }); }
const char* Torch_IValue_IsList              (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isList();               }); }
const char* Torch_IValue_IsBoolList          (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isBoolList();           }); }
const char* Torch_IValue_IsIntList           (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isIntList();            }); }
const char* Torch_IValue_IsDoubleList        (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isDoubleList();         }); }
const char* Torch_IValue_IsComplexDoubleList (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isComplexDoubleList();  }); }
const char* Torch_IValue_IsTensorList        (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isTensorList();         }); }
const char* Torch_IValue_IsGenericDict       (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isGenericDict();        }); }
const char* Torch_IValue_IsDevice            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isDevice();             }); }
const char* Torch_IValue_IsStorage           (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isStorage();            }); }
const char* Torch_IValue_IsCapsule           (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isCapsule();            }); }
const char* Torch_IValue_IsCustomClass       (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isCustomClass();        }); }
const char* Torch_IValue_IsFuture            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isFuture();             }); }
const char* Torch_IValue_IsRRef              (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isRRef();               }); }
const char* Torch_IValue_IsQuantizer         (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isQuantizer();          }); }
// const char* Torch_IValue_IsSymInt            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isSymInt();             }); }
// const char* Torch_IValue_IsSymFloat          (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isSymFloat();           }); }
// const char* Torch_IValue_IsOptionalTensorList(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isOptionalTensorList(); }); }
const char* Torch_IValue_IsObject            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isObject();             }); }
const char* Torch_IValue_IsModule            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isModule();             }); }
const char* Torch_IValue_IsPyObject          (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isPyObject();           }); }
const char* Torch_IValue_IsEnum              (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isEnum();               }); }
const char* Torch_IValue_IsStream            (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isStream();             }); }
const char* Torch_IValue_IsGenerator         (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isGenerator();          }); }
const char* Torch_IValue_IsPtrType           (bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->isPtrType();            }); }

// MARK: Container length checkers

const char* Torch_IValue_LengthTuple(int64_t* output, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        *output = ivalue->toTupleRef().size();
    });
}

const char* Torch_IValue_LengthList(int64_t* output, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        *output = ivalue->toListRef().size();
    });
}

const char* Torch_IValue_LengthDict(int64_t* output, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        *output = ivalue->toGenericDict().size();
    });
}

// MARK: Data accessors

template<typename T, typename U>
void throw_runtime_error_on_inequal_array_lengths(const T& expected, const U& actual) {
    if (expected == actual) return;
    throw std::runtime_error(
        "Expected input array of size " + std::to_string(expected) +
        " but received array of size " + std::to_string(actual)
    );
}

const char* Torch_IValue_ToNone(char** output, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto str = ivalue->toNone();
        *output = reinterpret_cast<char*>(malloc(str.size() + 1));
        snprintf(*output, str.size() + 1, "%s", str.c_str());
    });
}
// const char* Torch_IValue_ToScalar(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toScalar(); }); }
const char* Torch_IValue_ToBool(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toBool(); }); }
const char* Torch_IValue_ToInt(int* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toInt(); }); }
const char* Torch_IValue_ToDouble(double* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toDouble(); }); }
const char* Torch_IValue_ToComplexDouble(double _Complex* output, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto value = ivalue->toComplexDouble();
        *output = reinterpret_cast<double _Complex(&)>(value);
    });
}
const char* Torch_IValue_ToString(char** output, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto str = ivalue->toStringRef().c_str();
        auto length = strlen(str);
        *output = reinterpret_cast<char*>(malloc(length + 1));
        snprintf(*output, length + 1, "%s", str);
    });
}
const char* Torch_IValue_ToTensor(Tensor* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = new torch::Tensor(ivalue->toTensor()); }); }

const char* Torch_IValue_ToBoolList(bool* output, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto values = ivalue->toBoolList();
        throw_runtime_error_on_inequal_array_lengths(num_datums, values.size());
        for (int i = 0; i < values.size(); i++) output[i] = values[i];
    });
}

const char* Torch_IValue_ToIntList(int* output, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto values = ivalue->toIntList();
        throw_runtime_error_on_inequal_array_lengths(num_datums, values.size());
        for (int i = 0; i < values.size(); i++) output[i] = values[i];
    });
}

const char* Torch_IValue_ToDoubleList(double* output, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto values = ivalue->toDoubleList();
        throw_runtime_error_on_inequal_array_lengths(num_datums, values.size());
        for (int i = 0; i < values.size(); i++) output[i] = values[i];
    });
}

const char* Torch_IValue_ToComplexDoubleList(double _Complex* output, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto values = ivalue->toComplexDoubleList();
        throw_runtime_error_on_inequal_array_lengths(num_datums, values.size());
        for (int i = 0; i < values.size(); i++) {
            auto value = values.begin() + i;
            output[i] = reinterpret_cast<double _Complex(&)>(value);
        }
    });
}

const char* Torch_IValue_ToTensorList(Tensor* output, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto values = ivalue->toTensorVector();
        throw_runtime_error_on_inequal_array_lengths(num_datums, values.size());
        for (int i = 0; i < values.size(); i++)
            output[i] = new torch::Tensor(values.at(i));
    });
}

// const char* Torch_IValue_ToOptionalTensorList(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toOptionalTensorList(); }); }

const char* Torch_IValue_ToList(IValue* output, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto values = ivalue->toListRef();
        throw_runtime_error_on_inequal_array_lengths(num_datums, values.size());
        for (int i = 0; i < values.size(); i++)
            output[i] = new torch::IValue(values.at(i));
    });
}

const char* Torch_IValue_ToTuple(IValue* output, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto values = ivalue->toTupleRef();
        throw_runtime_error_on_inequal_array_lengths(num_datums, values.size());
        auto elements = values.elements();
        for (int i = 0; i < values.size(); i++)
            output[i] = new torch::IValue(elements.at(i));
    });
}

const char* Torch_IValue_ToGenericDict(IValue* keys, IValue* values, int64_t num_datums, IValue ivalue) {
    return try_catch_return_error_string([&]() {
        auto map = ivalue->toGenericDict();
        throw_runtime_error_on_inequal_array_lengths(num_datums, map.size());
        int i = 0;
        for (const auto& pair : map) {
            keys[i] = new torch::IValue(pair.key());
            values[i] = new torch::IValue(pair.value());
            i = i + 1;
        }
    });
}

const char* Torch_IValue_ToDevice(Device* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = new torch::Device(ivalue->toDevice()); }); }

// const char* Torch_IValue_ToFuture(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toFuture(); }); }
// const char* Torch_IValue_ToRRef(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toRRef(); }); }
// const char* Torch_IValue_ToQuantizer(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toQuantizer(); }); }
// const char* Torch_IValue_ToSymInt(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toSymInt(); }); }
// const char* Torch_IValue_ToSymFloat(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toSymInt(); }); }
// const char* Torch_IValue_ToStorage(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toStorage(); }); }
// const char* Torch_IValue_ToCapsule(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toCapsule(); }); }
// const char* Torch_IValue_ToCustomClass(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toCustomClass(); }); }
// const char* Torch_IValue_ToObject(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toObject(); }); }
// const char* Torch_IValue_ToModule(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toModule(); }); }
// const char* Torch_IValue_ToPyObject(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toPyObject(); }); }
// const char* Torch_IValue_ToEnum(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toEnum(); }); }
// const char* Torch_IValue_ToStream(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toStream(); }); }
// const char* Torch_IValue_ToGenerator(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toGenerator(); }); }
// const char* Torch_IValue_ToPtrType(bool* output, IValue ivalue) { return try_catch_return_error_string([&]() { *output = ivalue->toPtrType(); }); }
