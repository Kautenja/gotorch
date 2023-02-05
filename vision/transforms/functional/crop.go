// GoTorch port of torchvision.transforms.functional.crop.
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
	F "github.com/Kautenja/gotorch/nn/functional"
)

// Crop a slice from a tensor with shape (..., H, W).
// Crop the given image at specified location and output size. The input is a
// Tensor with expected shape of (..., H, W). If image size is smaller than
// output size along any edge, image is padded with 0 and then cropped.
func Crop(tensor torch.Tensor, xmin, ymin, xmax, ymax int64) torch.Tensor {
	shape := tensor.Shape()
	dim := len(shape)
	if dim < 2 { panic("Crop requires inputs with 2 or more dimensions") }
	yDim := int64(dim - 2)
	xDim := int64(dim - 1)
	H := shape[yDim]
	W := shape[xDim]
	// First check if the crop expands past the height or width of the tensor.
	var padx1 int64 = 0
	if xmax > W {
		padx1 = xmax - W
	}
	var pady1 int64 = 0
	if ymax > H {
		pady1 = ymax - H
	}
	// Next check if the crop expands past the (x,y) origin. This is done second
	// because it will influence the (xmax,ymax) indexes when padding the origin
	// due to a global shift of the pixel index grid.
	var padx0 int64 = 0
	if xmin < 0 {
		padx0 = -xmin
		xmax = xmax + padx0  // correct for the index grid shift
		xmin = 0             // Once padded, the index implicitly becomes 0
	}
	var pady0 int64 = 0
	if ymin < 0 {
		pady0 = -ymin
		ymax = ymax + pady0  // correct for the index grid shift
		ymin = 0             // Once padded, the index implicitly becomes 0
	}
	tensor = F.Pad(tensor, []int64{padx0, padx1, pady0, pady1}, F.PadConstant, 0)
	tensor = tensor.Slice(yDim, ymin, ymax, 1)  // t = t[..., ymin:ymax, :]
	tensor = tensor.Slice(xDim, xmin, xmax, 1)  // t = t[..., :, xmin:xmax]
	return tensor
}
