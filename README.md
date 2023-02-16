<p align="center">
<img
    src="https://user-images.githubusercontent.com/2184469/211366276-5e1b5bf2-0ec6-48e8-9718-679992809c35.png"
    width="80%"
    alt="GoTorch"
/>
</p>

[![GoTorch](https://godoc.org/github.com/narqo/go-badge?status.svg)](https://pkg.go.dev/github.com/Kautenja/gotorch)
[![Go Report Card](https://goreportcard.com/badge/github.com/Kautenja/gotorch)](https://goreportcard.com/report/github.com/Kautenja/gotorch)

GoTorch is a Golang front-end for [pytorch](https://github.com/pytorch/pytorch).

## Installation

GoTorch requires the installation of libtorch and libcgotorch (a C-binding
written for CGo.) To install these libraries to `/usr/local`, use:

```shell
git clone --depth 1 --branch v1.11.0-0.1.2 https://github.com/Kautenja/gotorch /tmp/gotorch
/tmp/gotorch/install.sh
rm -rf /tmp/gotorch
```

You may refer to the 
[example project](https://github.com/Kautenja/gotorch-example) for a working
demonstration of this library.
