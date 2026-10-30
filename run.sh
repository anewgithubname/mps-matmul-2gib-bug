#!/bin/sh
# Build and run the reproduction: ./run.sh [M N K]
set -e
cd "$(dirname "$0")"
out="${TMPDIR:-/tmp}/mps_matmul_2gib"
build() {
    clang++ -std=c++17 -O2 -fobjc-arc "$@" mps_matmul_2gib.mm \
        -framework Foundation -framework Metal -framework MetalPerformanceShaders -o "$out"
}
# Some Command Line Tools SDKs fail to link; fall back to the Xcode SDK.
build 2>/dev/null || build -isysroot "$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
exec "$out" "$@"
