// Which aligned GPU addresses trigger the MPSMatrixMultiplication NaNs?
//
// Allocates large, untouched shared buffers to walk the GPU address space. For
// each power-of-two alignment level found (an address that is a multiple of
// 2^p but not of 2^(p+1), p >= 30), it places the result matrix so that it
// straddles that address, runs the same GEMM, and counts NaNs and mismatches
// against a reference result computed far from any 1 GiB boundary.
//
// Build: clang++ -std=c++17 -O2 -fobjc-arc mps_matmul_align.mm \
//            -framework Foundation -framework Metal -framework MetalPerformanceShaders \
//            -o mps_matmul_align
// Run:   ./mps_matmul_align [M N K [maxBuffers]]   (default 6000 2500 64 48)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static const uint64_t kGiB = 1ull << 30;
static const uint64_t kMiB = 1ull << 20;

struct Site {
    id<MTLBuffer> buffer;
    uint64_t offset;    // result matrix offset in the buffer
    uint64_t boundary;  // the aligned GPU address it straddles
};

int main(int argc, char** argv) {
    const NSUInteger M = argc > 3 ? atoi(argv[1]) : 6000;
    const NSUInteger N = argc > 3 ? atoi(argv[2]) : 2500;
    const NSUInteger K = argc > 3 ? atoi(argv[3]) : 64;
    const int maxBuffers = argc > 4 ? atoi(argv[4]) : 48;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const MTLResourceOptions options = MTLResourceStorageModeShared;
    const NSUInteger rowBytes = N * sizeof(float), resultBytes = M * rowBytes;

    id<MTLBuffer> A = [device newBufferWithLength:M * K * sizeof(float) options:options];
    id<MTLBuffer> B = [device newBufferWithLength:K * N * sizeof(float) options:options];
    srand(1);
    for (NSUInteger i = 0; i < M * K; ++i) ((float*)A.contents)[i] = rand() / (float)RAND_MAX;
    for (NSUInteger i = 0; i < K * N; ++i) ((float*)B.contents)[i] = rand() / (float)RAND_MAX;

    auto matrix = [](id<MTLBuffer> b, uint64_t offset, NSUInteger rows, NSUInteger cols) {
        MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                      rowBytes:cols * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:b offset:offset descriptor:d];
    };
    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
        transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
    auto run = [&](id<MTLBuffer> C, uint64_t offset) {
        memset((char*)C.contents + offset, 0, resultBytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [gemm encodeToCommandBuffer:cb leftMatrix:matrix(A, 0, M, K) rightMatrix:matrix(B, 0, K, N)
                       resultMatrix:matrix(C, offset, M, N)];
        [cb commit];
        [cb waitUntilCompleted];
        return (const float*)((const char*)C.contents + offset);
    };

    // Reference: a result at least 16 MiB away from any 1 GiB boundary.
    std::vector<id<MTLBuffer>> keep;
    id<MTLBuffer> ref = nil;
    for (int i = 0; i < 64 && !ref; ++i) {
        id<MTLBuffer> b = [device newBufferWithLength:resultBytes options:options];
        const uint64_t s = b.gpuAddress, e = s + resultBytes + 16 * kMiB;
        if (s / kGiB == e / kGiB && s % kGiB >= 16 * kMiB) ref = b;
        else keep.push_back(b);
    }
    if (!ref) {
        printf("could not place a reference result away from 1 GiB boundaries\n");
        return 2;
    }
    const float* r = run(ref, 0);
    std::vector<float> expect(r, r + M * N);

    // Walk the address space with large untouched buffers and record, for each
    // alignment level, the first aligned address a result can straddle.
    const uint64_t bufBytes = 2 * kGiB;
    const uint64_t before = (M / 2) * rowBytes + rowBytes / 2;  // boundary lands mid-row
    std::map<int, Site> sites;
    uint64_t lo = UINT64_MAX, hi = 0;
    int allocated = 0;
    for (; allocated < maxBuffers; ++allocated) {
        id<MTLBuffer> b = [device newBufferWithLength:bufBytes options:options];
        if (!b) break;
        keep.push_back(b);
        const uint64_t base = b.gpuAddress;
        lo = std::min(lo, base);
        hi = std::max(hi, base + bufBytes);
        for (uint64_t x = (base + before + kGiB - 1) / kGiB * kGiB; x - before + resultBytes <= base + bufBytes;
             x += kGiB) {
            const int p = __builtin_ctzll(x);
            if (!sites.count(p)) sites[p] = {b, (x - before - base) / 256 * 256, x};
        }
    }

    printf("device %s, C = A(%lux%lu) * B(%lux%lu), result %.1f MB\n", device.name.UTF8String,
           (unsigned long)M, (unsigned long)K, (unsigned long)K, (unsigned long)N, resultBytes / 1e6);
    printf("walked %d buffers of 2 GiB, GPU addresses 0x%llx..0x%llx\n", allocated, lo, hi);
    printf("each result straddles one aligned address; the reference is far from any 1 GiB boundary\n\n");

    bool failed = false;
    for (auto& [p, s] : sites) {
        const float* c = run(s.buffer, s.offset);
        size_t nans = 0, mismatches = 0, firstNaN = SIZE_MAX, lastNaN = 0;
        for (NSUInteger i = 0; i < M * N; ++i) {
            if (std::isnan(c[i])) {
                ++nans;
                firstNaN = std::min<size_t>(firstNaN, i);
                lastNaN = i;
            } else if (c[i] != expect[i]) {
                ++mismatches;
            }
        }
        const uint64_t start = s.buffer.gpuAddress + s.offset;
        printf("2^%-2d  boundary 0x%011llx  low 32 bits 0x%08llx -> 0x%08llx  result [0x%llx, 0x%llx)  NaNs: %-8zu "
               "other mismatches: %zu",
               p, s.boundary, (s.boundary - 1) & 0xffffffffull, s.boundary & 0xffffffffull, start,
               start + resultBytes, nans, mismatches);
        if (nans) {
            uint32_t bits;
            memcpy(&bits, &c[firstNaN], 4);
            printf("  rows %zu..%zu, NaN bits 0x%08x", firstNaN / N, lastNaN / N, bits);
        }
        printf("\n");
        failed |= nans || mismatches;
    }
    if (sites.empty()) printf("no aligned address found\n");

    printf(failed ? "\nBUG REPRODUCED\n" : "\nno corruption observed\n");
    return failed ? 1 : 0;
}
