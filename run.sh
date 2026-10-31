#!/bin/sh
# Build and run a reproduction: ./run.sh [near] [M N K]
#   ./run.sh         result buffers inside a 2 GiB window and across a boundary
#   ./run.sh near    result buffers that end just below a boundary
set -e
cd "$(dirname "$0")"
prog=mps_matmul_2gib
if [ "$1" = near ]; then
    prog=mps_matmul_near_2gib
    shift
fi
out="${TMPDIR:-/tmp}/$prog"
build() {
    clang++ -std=c++17 -O2 -fobjc-arc "$@" $prog.mm \
        -framework Foundation -framework Metal -framework MetalPerformanceShaders -o "$out"
}
# Some Command Line Tools SDKs fail to link; fall back to the Xcode SDK.
build 2>/dev/null || build -isysroot "$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
exec "$out" "$@"
