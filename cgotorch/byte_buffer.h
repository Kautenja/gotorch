// An abstraction of a buffer of bytes.
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

/// @brief Return a void pointer to the byte buffer's data.
/// @param buffer The byte buffer to return a pointer to the data of.
/// @returns A pointer to the raw data buffer.
void *ByteBuffer_Data(ByteBuffer buffer);

/// @brief Return the number of bytes in the byte buffer.
/// @param buffer The byte buffer to return the size of.
/// @returns The number of bytes contained byte the buffer.
int64_t ByteBuffer_Size(ByteBuffer buffer);

/// @brief Free a byte buffer from the heap.
/// @param buffer The byte buffer to free.
void ByteBuffer_Free(ByteBuffer buffer);

#ifdef __cplusplus
}
#endif
