# GoTorch

A Golang front-end for
[libtorch](https://pytorch.org/cppdocs/api/library_root.html).

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
