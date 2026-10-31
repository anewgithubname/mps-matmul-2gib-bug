// Which GPU kernel does MPSMatrixMultiplication run on this machine?
//
// Swizzles the concrete compute encoder class and prints every pipeline,
// buffer binding, inline constant and dispatch that one GEMM encodes. The
// pipeline label names the kernel.
//
// Build: clang++ -std=c++17 -O2 -fobjc-arc mps_gemm_trace.mm \
//            -framework Foundation -framework Metal -framework MetalPerformanceShaders \
//            -o mps_gemm_trace
// Run:   ./mps_gemm_trace [M N K]...
//        (default: 6000 2500 64, 4000 1000 32, 6000 2500 7, 6016 2560 64)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

static bool gTrace = false;

static void printSize(const char* what, MTLSize g, MTLSize t) {
    printf("    %s (%lu, %lu, %lu)  threadgroup (%lu, %lu, %lu)\n", what, (unsigned long)g.width,
           (unsigned long)g.height, (unsigned long)g.depth, (unsigned long)t.width, (unsigned long)t.height,
           (unsigned long)t.depth);
}

static void swizzleEncoderClass(Class cls) {
    static std::set<Class> done;
    if (!done.insert(cls).second) return;
    printf("encoder class %s\n", class_getName(cls));

    {
        SEL sel = @selector(setComputePipelineState:);
        Method m = class_getInstanceMethod(cls, sel);
        typedef void (*Fn)(id, SEL, id);
        Fn orig = (Fn)method_getImplementation(m);
        method_setImplementation(m, imp_implementationWithBlock(^(id self, id<MTLComputePipelineState> pso) {
            if (gTrace)
                printf("    pipeline %s  (max %lu threads, SIMD width %lu)\n", pso.label ? pso.label.UTF8String : "(no label)",
                       (unsigned long)pso.maxTotalThreadsPerThreadgroup, (unsigned long)pso.threadExecutionWidth);
            orig(self, sel, pso);
        }));
    }
    {
        SEL sel = @selector(setBuffer:offset:atIndex:);
        Method m = class_getInstanceMethod(cls, sel);
        typedef void (*Fn)(id, SEL, id, NSUInteger, NSUInteger);
        Fn orig = (Fn)method_getImplementation(m);
        method_setImplementation(m, imp_implementationWithBlock(^(id self, id<MTLBuffer> buf, NSUInteger off, NSUInteger idx) {
            if (gTrace)
                printf("    buffer[%lu] length %lu offset %lu\n", (unsigned long)idx, (unsigned long)buf.length,
                       (unsigned long)off);
            orig(self, sel, buf, off, idx);
        }));
    }
    {
        SEL sel = @selector(setBytes:length:atIndex:);
        Method m = class_getInstanceMethod(cls, sel);
        typedef void (*Fn)(id, SEL, const void*, NSUInteger, NSUInteger);
        Fn orig = (Fn)method_getImplementation(m);
        method_setImplementation(m, imp_implementationWithBlock(^(id self, const void* p, NSUInteger len, NSUInteger idx) {
            if (gTrace) {
                printf("    bytes[%lu] %lu B as uint32:", (unsigned long)idx, (unsigned long)len);
                for (NSUInteger i = 0; i < len / 4 && i < 32; ++i) printf(" %u", ((const uint32_t*)p)[i]);
                printf("\n");
            }
            orig(self, sel, p, len, idx);
        }));
    }
    {
        SEL sel = @selector(setThreadgroupMemoryLength:atIndex:);
        Method m = class_getInstanceMethod(cls, sel);
        typedef void (*Fn)(id, SEL, NSUInteger, NSUInteger);
        Fn orig = (Fn)method_getImplementation(m);
        method_setImplementation(m, imp_implementationWithBlock(^(id self, NSUInteger len, NSUInteger idx) {
            if (gTrace) printf("    threadgroup memory[%lu] %lu B\n", (unsigned long)idx, (unsigned long)len);
            orig(self, sel, len, idx);
        }));
    }
    {
        SEL sel = @selector(dispatchThreadgroups:threadsPerThreadgroup:);
        Method m = class_getInstanceMethod(cls, sel);
        typedef void (*Fn)(id, SEL, MTLSize, MTLSize);
        Fn orig = (Fn)method_getImplementation(m);
        method_setImplementation(m, imp_implementationWithBlock(^(id self, MTLSize g, MTLSize t) {
            if (gTrace) printSize("dispatch threadgroups", g, t);
            orig(self, sel, g, t);
        }));
    }
    {
        SEL sel = @selector(dispatchThreads:threadsPerThreadgroup:);
        Method m = class_getInstanceMethod(cls, sel);
        typedef void (*Fn)(id, SEL, MTLSize, MTLSize);
        Fn orig = (Fn)method_getImplementation(m);
        method_setImplementation(m, imp_implementationWithBlock(^(id self, MTLSize g, MTLSize t) {
            if (gTrace) printSize("dispatch threads", g, t);
            orig(self, sel, g, t);
        }));
    }
}

int main(int argc, char** argv) {
    std::vector<std::array<NSUInteger, 3>> shapes;
    for (int i = 1; i + 2 < argc; i += 3)
        shapes.push_back({(NSUInteger)atoi(argv[i]), (NSUInteger)atoi(argv[i + 1]), (NSUInteger)atoi(argv[i + 2])});
    if (shapes.empty()) shapes = {{6000, 2500, 64}, {4000, 1000, 32}, {6000, 2500, 7}, {6016, 2560, 64}};

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    printf("device %s, macOS %s\n", device.name.UTF8String,
           [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);

    // MPS may ask for either dispatch type; hook the encoder class behind each.
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        swizzleEncoderClass([e class]);
        [e endEncoding];
        e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        swizzleEncoderClass([e class]);
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    const MTLResourceOptions options = MTLResourceStorageModeShared;
    auto matrix = [](id<MTLBuffer> b, NSUInteger rows, NSUInteger cols) {
        MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                      rowBytes:cols * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:b descriptor:d];
    };
    for (auto [M, N, K] : shapes) {
        id<MTLBuffer> A = [device newBufferWithLength:M * K * sizeof(float) options:options];
        id<MTLBuffer> B = [device newBufferWithLength:K * N * sizeof(float) options:options];
        id<MTLBuffer> C = [device newBufferWithLength:M * N * sizeof(float) options:options];
        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
            transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
        printf("\nC = A(%lux%lu) * B(%lux%lu)\n", (unsigned long)M, (unsigned long)K, (unsigned long)K, (unsigned long)N);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        gTrace = true;
        [gemm encodeToCommandBuffer:cb leftMatrix:matrix(A, M, K) rightMatrix:matrix(B, K, N) resultMatrix:matrix(C, M, N)];
        gTrace = false;
        [cb commit];
        [cb waitUntilCompleted];
    }
    return 0;
}
