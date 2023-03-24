package main

import (
	torch "github.com/Kautenja/gotorch"
	F "github.com/Kautenja/gotorch/nn/functional"
	// T "github.com/Kautenja/gotorch/vision/transforms/functional"
)

func main() {
	a := torch.Rand([]int64{3, 256, 256}, torch.NewTensorOptions())
	_ = a.Shape()
	a = a.Unsqueeze(0).CastTo(torch.Float)
	a = F.InterpolateSize(a, []int64{128, 128}, F.InterpolateBilinear, true, false)


	// device := torch.NewDevice("cpu")
	// _ = torch.Rand([]int64{3, 256, 256}, torch.NewTensorOptions()).CopyTo(device).Unsqueeze(0)
}
