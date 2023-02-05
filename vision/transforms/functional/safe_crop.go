// A cropping method that safely extends the image window.
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

package vision_transforms_functional

import (
	"github.com/Kautenja/gotorch"
)

// Crop a slice from a tensor with shape (..., H, W).
//
// Bounding boxes are in (xmin,ymin,xmax,ymax) format.
//
// The python syntax for this function would be:
// ```python
// tensor[..., ymin:ymax, xmin:ymax]
// ```
//
// Note that the semantics of this function are (xmin,ymax,xmax,ymax), whereas
// the semantics for PyTorch torchvision is actually (ymin,xmin,height,width.)
// The semantics for edge cases are to clip to the bounds whereas in PyTorch
// the conventions are allow shifts past the window and larger bounds via
// zero padding.
//
func SafeCrop(tensor torch.Tensor, xmin, ymin, xmax, ymax int64) torch.Tensor {
	shape := tensor.Shape()
	dim := len(shape)
	if dim < 2 { panic("Crop requires inputs with 2 or more dimensions") }
	H := shape[int64(dim - 2)]
	W := shape[int64(dim - 1)]
	if xmin < 0 { xmin = 0 }
	if ymin < 0 { ymin = 0 }
	if xmax > W { xmax = W }
	if ymax > H { ymax = H }
	tensor = tensor.Slice(int64(dim - 2), ymin, ymax, 1)  // t = t[..., ymin:ymax, :]
	tensor = tensor.Slice(int64(dim - 1), xmin, xmax, 1)  // t = t[..., :, xmin:xmax]
	return tensor
}
