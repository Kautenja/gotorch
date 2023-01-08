// GoTorch port of torchvision.transforms.Normalize.
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

package vision_transforms

import (
    "fmt"
    "github.com/Kautenja/gotorch"
)

// A Normalize transformer that stores channel-wise mean and standard deviation.
type NormalizeTransformer struct {
    mean, stddev torch.Tensor
}

// Create a new NormalizeTransformer with given mean and standard deviation.
func Normalize(mean, stddev []float32) *NormalizeTransformer {
    if len(mean) == 0 { panic("len(mean) should be greater than 0") }
    if len(stddev) == 0 { panic("len(stddev) should be greater than 0") }
    if len(mean) != len(stddev) {
        panic(fmt.Sprintf("len(mean)=%d and len(stddev)=%d should be the same", len(mean), len(stddev)))
    }
    // Create the mean and stddev tensors in (1, C, 1, 1) float format.
    transformer := NormalizeTransformer{
        mean:   torch.NewTensor(mean).Unsqueeze(0).Unsqueeze(-1).Unsqueeze(-1),
        stddev: torch.NewTensor(stddev).Unsqueeze(0).Unsqueeze(-1).Unsqueeze(-1),
    }
    // Preemptively check for divide-by-zero errors in the forward pass.
    if transformer.stddev.Eq(torch.ZerosLike(transformer.stddev)).Any().Item().(bool) {
        panic("stddev contains zeros (pre-emptive divide-by-zero error)")
    }
    return &transformer
}

// Forward pass an image through the transformer to map its data to N(0, 1).
func (t NormalizeTransformer) Forward(input torch.Tensor) torch.Tensor {
    return input.Sub(t.mean, 1.0).Div(t.stddev)
}
