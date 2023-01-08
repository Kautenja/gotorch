// A function for converting exception error handling into char* conventions.
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

#include <stdlib.h>   // for malloc
#include <stdio.h>    // for snprintf
#include <string.h>   // for strlen
#include <exception>

#ifdef __cplusplus

/// @brief Try a function and return a new error string if an error occurs.
/// @param callback The callback that might throw a `std::exception`.
/// @returns A pointer to the error string.
/// @details
/// Ownership of the string output is transferred to the caller. If the output
/// not nil, it should be freed by the caller when handling the error.
template<typename T>
inline const char* try_catch_return_error_string(const T& callback) {
    try {
        callback();
        return nullptr;
    } catch (const std::exception &error) {
        const char* error_message = error.what();
        std::size_t length = strlen(error_message);
        char* output = reinterpret_cast<char*>(malloc(length + 1));
        snprintf(output, length + 1, "%s", error_message);
        return output;
    }
}

#endif
