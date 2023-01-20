#!/bin/sh
#
# Usage main.sh [options] [routine]
#
# Options:
#     -h       Print documentation about this script
#
# Commands:
#     dockerbuild  Build a development environment in a Docker container
#     init         Initialize the go project (download requirements, cleanup, etc.)
#     makedata     Generate data for testing.
#     build        Build the C code for the current platform
#     test         Execute the Golang unit test suite for the project in
#     coverage     Generate an HTML coverage report
#

# --- Constants --------------------------------------------------------------

export ENV_FILE_PATH="${PWD}/.env"
export OPERATING_SYSTEM=$(uname -s)
VERSION_FILE="${PWD}/pkg/version/version.go"

# --- Functions --------------------------------------------------------------

# Print the help string at the top of this script.
# Reference: https://josh.fail/2019/how-to-print-help-text-in-shell-scripts/
print_help() {
  sed -ne '/^#/!q;s/.\{1,2\}//;1d;p' < "$0"
}

# --- Options processing -----------------------------------------------------

CONTAINER=0

while getopts ":dhl" optname; do
  case "$optname" in
  "d")
    echo "Running operation within container"
    CONTAINER=1
    ;;
  "h")
    print_help
    exit 1
    ;;
  "?")
    echo "Unknown option $OPTARG"
    exit 1
    ;;
  ":")
    echo "No argument value for option $OPTARG"
    exit 1
    ;;
  *)
    echo "Unknown error while processing options"
    exit 1
    ;;
  esac
done

_timeout() { ( set +b; sleep "$1" & "${@:2}" & wait; r=$?; kill -9 `jobs -p`; exit $r; ) }

shift $(($OPTIND - 1))

# --- Body -------------------------------------------------------------------

IMAGE=sensory-cgotorch

case "$1" in

"")
  print_help
  exit 0;
;;

"dockerbuild")
  docker build -t ${IMAGE} . || exit 1
  exit 0;
;;

"init")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh init"
    exit 0;
  fi
  go mod download -x
  go mod tidy
  exit 0;
;;

"virtualenv")
  cd ./scripts
  virtualenv -p python3 .env
  source .env/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  exit 0;
;;

"makedata")
  mkdir -p ./data
  cd ./scripts
  source .env/bin/activate
  for script in *.py; do python "$script"; done
  exit 0;
;;

"build")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh build"
    exit 0;
  fi
  go generate
  go build
  exit 0;
;;

"test")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh test"
    exit 0;
  fi
  go test -cover ./...
  exit 0;
;;

"test_verbose")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh test"
    exit 0;
  fi
  go test -v -cover ./...
  exit 0;
;;

"coverage")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh coverage"
    exit 0;
  fi
  go test -coverprofile=coverage.out ./...
  go tool cover -html=coverage.out
  exit 0;
;;

"ci")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh ci"
    exit 0;
  fi
  $0 init
  $0 build
  $0 test
  exit 0;
;;

*)
  echo "Unknown command: $1"
  exit 0;
;;

esac
