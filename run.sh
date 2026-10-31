#!/bin/sh
# Build and run a reproduction: ./run.sh [near|align|trace] [M N K]
#   ./run.sh         result buffers inside a 2 GiB window and across a boundary
#   ./run.sh near    result buffers that end just below a boundary
#   ./run.sh align   results across 1 GiB, 2 GiB, 4 GiB, ... aligned addresses
#   ./run.sh trace   the GPU kernels MPSMatrixMultiplication runs (several M N K allowed)
set -e
cd "$(dirname "$0")"
prog=mps_matmul_2gib
case "$1" in
near) prog=mps_matmul_near_2gib; shift ;;
align) prog=mps_matmul_align; shift ;;
trace) prog=mps_gemm_trace; shift ;;
esac
out="${TMPDIR:-/tmp}/$prog"
build() {
    clang++ -std=c++17 -O2 -fobjc-arc "$@" $prog.mm \
        -framework Foundation -framework Metal -framework MetalPerformanceShaders -o "$out"
}
# Some Command Line Tools SDKs fail to link; fall back to the Xcode SDK.
build 2>/dev/null || build -isysroot "$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
exec "$out" "$@"
