// Minimal reproduction: MPSMatrixMultiplication writes NaNs into its result
// when the result buffer straddles a 2 GiB-aligned GPU virtual address.
//
// Build: clang++ -std=c++17 -O2 -fobjc-arc mps_matmul_2gib.mm \
//            -framework Foundation -framework Metal -framework MetalPerformanceShaders \
//            -o mps_matmul_2gib
// Run:   ./mps_matmul_2gib [M N K]      (default 6000 2500 64)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

static const uint64_t kTwoGiB = 1ull << 31;

static bool crosses2GiB(id<MTLBuffer> b, uint64_t margin = 0) {
    return b.gpuAddress / kTwoGiB != (b.gpuAddress + b.length + margin - 1) / kTwoGiB;
}

// A buffer that ends less than about 8 MiB below a boundary is also corrupted
// (see mps_matmul_near_2gib.mm), so the reference buffer keeps a wider margin.
static const uint64_t kMargin = 16ull << 20;

static bool tty = false;

// One map cell: red '#' if any element in rows [r0, r1) x cols [c0, c1) is NaN, else '.'.
static void putCell(const float* c, NSUInteger N, NSUInteger r0, NSUInteger r1, NSUInteger c0, NSUInteger c1) {
    bool nan = false;
    for (NSUInteger r = r0; r < r1 && !nan; ++r)
        for (NSUInteger j = c0; j < c1 && !nan; ++j) nan = std::isnan(c[r * N + j]);
    printf(nan ? (tty ? "\033[41m#\033[0m" : "#") : ".");
}

// Both results side by side, each squeezed into a grid of cells.
static void drawMaps(const float* inside, const float* straddling, NSUInteger M, NSUInteger N, NSUInteger boundaryRow) {
    const NSUInteger lines = 30, width = 36;
    printf("\n%-15s %-*s   %s\n", "", (int)width + 2, "inside (correct)", "straddling ('#' = NaN)");
    for (NSUInteger y = 0; y < lines; ++y) {
        const NSUInteger r0 = M * y / lines, r1 = M * (y + 1) / lines;
        printf("rows %4lu-%4lu  |", (unsigned long)r0, (unsigned long)r1 - 1);
        for (NSUInteger x = 0; x < width; ++x) putCell(inside, N, r0, r1, N * x / width, N * (x + 1) / width);
        printf("|   |");
        for (NSUInteger x = 0; x < width; ++x) putCell(straddling, N, r0, r1, N * x / width, N * (x + 1) / width);
        printf("|%s\n", r0 <= boundaryRow && boundaryRow < r1 ? " <- 2 GiB boundary" : "");
    }
}

// The top-left corner of the NaN band, element by element.
static void drawZoom(const float* c, NSUInteger N, NSUInteger firstRow) {
    printf("\nstraddling result, rows %lu-%lu, columns 0-9:\n", (unsigned long)firstRow - 2, (unsigned long)firstRow + 2);
    for (NSUInteger r = firstRow - 2; r <= firstRow + 2; ++r) {
        printf("row %5lu ", (unsigned long)r);
        for (NSUInteger j = 0; j < 10; ++j) {
            const float v = c[r * N + j];
            if (std::isnan(v)) printf(tty ? " \033[41m    NaN\033[0m" : "     NaN");
            else printf(" %7.3f", v);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    tty = isatty(1);
    const NSUInteger M = argc > 3 ? atoi(argv[1]) : 6000;
    const NSUInteger N = argc > 3 ? atoi(argv[2]) : 2500;
    const NSUInteger K = argc > 3 ? atoi(argv[3]) : 64;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    auto newBuffer = [&](NSUInteger bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    };

    // Inputs: ordinary values in [0, 1].
    id<MTLBuffer> A = newBuffer(M * K * sizeof(float));
    id<MTLBuffer> B = newBuffer(K * N * sizeof(float));
    srand(1);
    for (NSUInteger i = 0; i < M * K; ++i) ((float*)A.contents)[i] = rand() / (float)RAND_MAX;
    for (NSUInteger i = 0; i < K * N; ++i) ((float*)B.contents)[i] = rand() / (float)RAND_MAX;

    // Two result buffers of the same size: one inside a 2 GiB-aligned window
    // and at least kMargin below its end, one straddling a 2 GiB boundary.
    // Allocate result-sized buffers until both exist. The extra buffers are
    // never touched, so they cost address space only.
    const NSUInteger resultBytes = M * N * sizeof(float);
    id<MTLBuffer> inside = nil, straddling = nil;
    std::vector<id<MTLBuffer>> padding;
    for (int i = 0; i < 4096 && !(inside && straddling); ++i) {
        id<MTLBuffer> b = newBuffer(resultBytes);
        if (!b) break;
        if (crosses2GiB(b) && !straddling) straddling = b;
        else if (!crosses2GiB(b, kMargin) && !inside) inside = b;
        else padding.push_back(b);
    }
    if (!inside || !straddling) {
        printf("could not place a result buffer across a 2 GiB boundary\n");
        return 2;
    }

    auto matrix = [](id<MTLBuffer> b, NSUInteger rows, NSUInteger cols) {
        MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                      rowBytes:cols * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:b descriptor:d];
    };
    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
        transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

    // The same C = A * B into each result buffer.
    for (id<MTLBuffer> C : {inside, straddling}) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [gemm encodeToCommandBuffer:cb leftMatrix:matrix(A, M, K) rightMatrix:matrix(B, K, N)
                       resultMatrix:matrix(C, M, N)];
        [cb commit];
        [cb waitUntilCompleted];
    }

    printf("device %s, C = A(%lux%lu) * B(%lux%lu), result %.1f MB\n", device.name.UTF8String,
           (unsigned long)M, (unsigned long)K, (unsigned long)K, (unsigned long)N, resultBytes / 1e6);
    const float* ref = (const float*)inside.contents;
    bool failed = false;
    size_t firstNaNRow = 0;
    for (id<MTLBuffer> C : {inside, straddling}) {
        const float* c = (const float*)C.contents;
        size_t nans = 0, mismatches = 0, firstNaN = SIZE_MAX, lastNaN = 0;
        for (NSUInteger i = 0; i < M * N; ++i) {
            if (std::isnan(c[i])) {
                ++nans;
                firstNaN = std::min<size_t>(firstNaN, i);
                lastNaN = i;
            } else if (c[i] != ref[i]) {
                ++mismatches;
            }
        }
        printf("%-10s gpuAddress [0x%llx, 0x%llx)  crosses 2 GiB: %-3s  NaNs: %zu  other mismatches: %zu",
               C == inside ? "inside" : "straddling", C.gpuAddress, C.gpuAddress + C.length,
               crosses2GiB(C) ? "yes" : "no", nans, mismatches);
        if (nans) {
            const uint64_t boundaryRow = ((C.gpuAddress / kTwoGiB + 1) * kTwoGiB - C.gpuAddress) / (N * sizeof(float));
            firstNaNRow = firstNaN / N;
            printf("\n           NaN rows %zu..%zu (the boundary falls in row %llu)", firstNaN / N, lastNaN / N,
                   (unsigned long long)boundaryRow);
        }
        printf("\n");
        failed |= nans || mismatches;
    }

    if (firstNaNRow >= 2) {
        drawMaps(ref, (const float*)straddling.contents, M, N,
                 ((straddling.gpuAddress / kTwoGiB + 1) * kTwoGiB - straddling.gpuAddress) / (N * sizeof(float)));
        drawZoom((const float*)straddling.contents, N, firstNaNRow);
        printf("\n");
    }

    // Sanity check on one element against a plain CPU dot product.
    double s = 0;
    for (NSUInteger k = 0; k < K; ++k) s += ((float*)A.contents)[k] * (double)((float*)B.contents)[k * N];
    printf("C[0,0]: CPU %.6f, inside %.6f\n", s, ref[0]);
    printf(failed ? "BUG REPRODUCED\n" : "no corruption observed\n");
    return failed ? 1 : 0;
}
