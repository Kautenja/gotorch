#!/bin/sh
#
# Usage main.sh [options] [routine]
#
# Options:
#     -h           Print documentation about this script
#     -v           Enable verbose output
#
# Commands:
#     dockerbuild  Build a development environment in a Docker container
#     virtualenv   Create a vritual Python environment for generating test data
#     makedata     Generate data for testing.
#     install      Download, compile, and install C code
#     download     Download go dependencies
#     tidy         Cleanup the go.mod and go.sum
#     build        Build the golang code
#     test         Execute the Golang unit test suite for the project in
#     coverage     Generate an HTML coverage report
#     ci           Install, download, tidy, build, and test
#     bash         Start a bash session within the container
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
VERBOSE=0

while getopts ":dhlv" optname; do
  case "$optname" in
  "d")
    echo "Running operation within container"
    CONTAINER=1
    ;;
  "v")
    VERBOSE=1
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

"install")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh install"
    exit 0;
  fi
  ./install.sh
  exit 0;
;;

"download")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh install"
    exit 0;
  fi
  go mod download -x
  exit 0;
;;

"tidy")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh install"
    exit 0;
  fi
  go mod tidy
  exit 0;
;;

"build")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh install"
    exit 0;
  fi
  go build
  exit 0;
;;

"test")
  if [ $CONTAINER -eq 1 ]; then
    docker run --rm ${IMAGE} bash -c "./main.sh test"
    exit 0;
  fi
  if [ $VERBOSE -eq 1 ]; then
    go test -v -cover ./...
  else
    go test -cover ./...
  fi
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
  $0 install
  $0 download
  $0 tidy
  $0 build
  $0 test
  exit 0;
;;

"bash")
  docker run --rm -it ${IMAGE} bash
  exit 0;
;;

*)
  echo "Unknown command: $1"
  exit 0;
;;

esac
