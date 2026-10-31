// Second reproduction: MPSMatrixMultiplication also writes NaNs into a result
// buffer that does not cross a 2 GiB-aligned GPU virtual address but ends a
// few MiB below one.
//
// A placement heap puts the same result buffer at exact distances below a
// boundary, and once starting at it. None of these buffers crosses it.
//
// Build: clang++ -std=c++17 -O2 -fobjc-arc mps_matmul_near_2gib.mm \
//            -framework Foundation -framework Metal -framework MetalPerformanceShaders \
//            -o mps_matmul_near_2gib
// Run:   ./mps_matmul_near_2gib [M N K]      (default 6000 2500 64)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const uint64_t kTwoGiB = 1ull << 31;
static const uint64_t kMiB = 1ull << 20;

int main(int argc, char** argv) {
    const NSUInteger M = argc > 3 ? atoi(argv[1]) : 6000;
    const NSUInteger N = argc > 3 ? atoi(argv[2]) : 2500;
    const NSUInteger K = argc > 3 ? atoi(argv[3]) : 64;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const MTLResourceOptions options = MTLResourceStorageModeShared;

    // Inputs: ordinary values in [0, 1].
    id<MTLBuffer> A = [device newBufferWithLength:M * K * sizeof(float) options:options];
    id<MTLBuffer> B = [device newBufferWithLength:K * N * sizeof(float) options:options];
    srand(1);
    for (NSUInteger i = 0; i < M * K; ++i) ((float*)A.contents)[i] = rand() / (float)RAND_MAX;
    for (NSUInteger i = 0; i < K * N; ++i) ((float*)B.contents)[i] = rand() / (float)RAND_MAX;

    // A placement heap with room for a result buffer ending 16 MiB below a
    // boundary and for one starting at it. Create heaps until one is placed
    // like that. The extra heaps are never touched.
    const NSUInteger resultBytes = M * N * sizeof(float);
    const MTLSizeAndAlign sa = [device heapBufferSizeAndAlignWithLength:resultBytes options:options];
    const uint64_t below = sa.size + 16 * kMiB + sa.align, above = sa.size;
    const uint64_t heapSize = below + above + 256 * kMiB;
    id<MTLHeap> heap = nil;
    uint64_t boundaryOffset = 0;
    std::vector<id<MTLHeap>> padding;
    for (int i = 0; i < 4096 && !heap; ++i) {
        MTLHeapDescriptor* d = [MTLHeapDescriptor new];
        d.type = MTLHeapTypePlacement;
        d.storageMode = MTLStorageModeShared;
        d.size = heapSize;
        id<MTLHeap> h = [device newHeapWithDescriptor:d];
        if (!h) break;
        const uint64_t base = [h newBufferWithLength:sa.align options:options offset:0].gpuAddress;
        const uint64_t toBoundary = (base / kTwoGiB + 1) * kTwoGiB - base;
        if (toBoundary >= below && toBoundary + above <= heapSize) {
            heap = h;
            boundaryOffset = toBoundary;
        } else {
            padding.push_back(h);
        }
    }
    if (!heap) {
        printf("could not place a heap across a 2 GiB boundary\n");
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

    printf("device %s, C = A(%lux%lu) * B(%lux%lu), result %.1f MB\n", device.name.UTF8String,
           (unsigned long)M, (unsigned long)K, (unsigned long)K, (unsigned long)N, resultBytes / 1e6);
    printf("each result buffer is placed alone; none crosses the 2 GiB boundary\n\n");

    // Heap offsets: the buffer ends this many MiB below the boundary, then one starting at it.
    std::vector<uint64_t> offsets;
    for (uint64_t gap : {16, 12, 8, 7, 6, 5, 4, 3, 2, 1, 0})
        offsets.push_back((boundaryOffset - gap * kMiB - resultBytes) / sa.align * sa.align);
    offsets.push_back(boundaryOffset);

    bool failed = false;
    for (uint64_t offset : offsets) {
        id<MTLBuffer> C = [heap newBufferWithLength:resultBytes options:options offset:offset];
        memset(C.contents, 0, resultBytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [gemm encodeToCommandBuffer:cb leftMatrix:matrix(A, M, K) rightMatrix:matrix(B, K, N)
                       resultMatrix:matrix(C, M, N)];
        [cb commit];
        [cb waitUntilCompleted];

        const float* c = (const float*)C.contents;
        size_t nans = 0;
        for (NSUInteger i = 0; i < M * N; ++i) nans += std::isnan(c[i]);
        const uint64_t start = C.gpuAddress, end = start + resultBytes;
        const uint64_t boundary = (start / kTwoGiB + 1) * kTwoGiB;
        char where[64];
        if (start % kTwoGiB == 0) snprintf(where, sizeof where, "starts at the boundary");
        else snprintf(where, sizeof where, "ends %6.3f MiB below the boundary", (boundary - end) / (double)kMiB);
        printf("gpuAddress [0x%llx, 0x%llx)  %-34s  NaNs: %zu\n", start, end, where, nans);
        failed |= nans > 0;
    }

    printf(failed ? "\nBUG REPRODUCED\n" : "\nno corruption observed\n");
    return failed ? 1 : 0;
}
