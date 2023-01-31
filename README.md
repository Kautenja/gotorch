<p align="center">
<img
    src="https://user-images.githubusercontent.com/2184469/211366276-5e1b5bf2-0ec6-48e8-9718-679992809c35.png"
    width="80%"
    alt="GoTorch"
/>
</p>

[![GoTorch](https://godoc.org/github.com/narqo/go-badge?status.svg)](https://pkg.go.dev/github.com/Kautenja/gotorch)

GoTorch is a Golang front-end for [pytorch](https://github.com/pytorch/pytorch).

## Installation

GoTorch requires the installation of libtorch and libcgotorch (a C-binding)
written for CGo. To install these libraries (to `/usr/local`,) use:

```shell
git clone --depth 1 --branch v1.11.0-0.0.0 https://github.com/Kautenja/gotorch
./gotorch/install.sh
rm -rf ./gotorch
```

You may refer to the 
[example project](https://github.com/Kautenja/gotorch-example) for a working
demonstration of this library.

## Development

##### Initialize

To initialize the go environment (e.g., download requirements, cleanup, etc.)

```shell
./main.sh init
```

##### Compile

To build the `libcgotorch` bridging library for `libtorch` and compile go code:

```shell
./main.sh build
```

##### Unit Test

To run test cases (after compiling code with `./main.sh build`):

```shell
./main.sh test
```

##### Continuous Integration

For convenience, the last three commands can be executed in series using:

```shell
./main.sh ci
```

## Docker Container

By default, development code is compiled directly for the host operating system
(assumed to be either MacOS or Linux.) Development functions can optionally be
executed within a Docker container using the `-d` flag. First, to build the
development Docker image:

```shell
./main.sh builddocker
```

For example, to run CI within the Docker container, use:

```shell
./main.sh -d ci
```
