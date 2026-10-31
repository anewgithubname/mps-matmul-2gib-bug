# MPSMatrixMultiplication writes NaNs into results near a 2 GiB GPU address

`MPSMatrixMultiplication` fills part of its result with NaNs when the result
spans a 2 GiB-aligned GPU virtual address, that is, when `gpuAddress / 2^31`
differs between the first and last byte of the result. It also does so when
the result does not cross such an address but ends less than about 8 MiB
below one. The same multiplication into an identical buffer away from a
boundary is correct. The input values and the result size do not matter: a
4 MB result is enough.

The bug is in one GPU kernel. On an Apple M2, `MPSMatrixMultiplication` runs
`MatrixMultiplyNNA14_M32_N128_singlebuffer_edgecase_all_types` for most
shapes, and only that kernel is affected. MPS picks other kernels, which are
not affected, when K is 7 or less, or when M and N are multiples of 64 and K is
a multiple of 16. An Apple M4 runs a different kernel for every shape and is not
affected. macOS 27.2 still has the bug.

We found no mention of this in Apple's documentation or developer forums.

## Test on your Mac

With the Command Line Tools installed (`xcode-select --install`), run:

```sh
git clone --depth 1 https://github.com/anewgithubname/mps-matmul-2gib-bug && cd mps-matmul-2gib-bug && ./run.sh; ./run.sh near; ./run.sh trace 6000 2500 64 | grep pipeline; sw_vers; sysctl -n hw.model
```

It takes a few seconds. Each of the first two programs ends with
`BUG REPRODUCED` or `no corruption observed`, and the last one prints the name
of the GPU kernel that MPS runs. Either way, please
[open an issue](https://github.com/anewgithubname/mps-matmul-2gib-bug/issues/new)
with the full output, so we can track which chips and macOS versions are
affected. Results from a virtual machine do not count: its GPU is
paravirtualized.

## Reproduce

```sh
./run.sh            # M N K default to 6000 2500 64
./run.sh 4000 1000 32
./run.sh near       # result buffers that end just below a boundary
./run.sh near 4000 1000 32
./run.sh align      # results across 1 GiB, 2 GiB, 4 GiB, ... aligned addresses
./run.sh trace      # the GPU kernels MPS runs; takes several M N K triples
```

[`run.sh`](run.sh) builds one program into `$TMPDIR` and runs it:
[`mps_matmul_2gib.mm`](mps_matmul_2gib.mm) by default,
[`mps_matmul_near_2gib.mm`](mps_matmul_near_2gib.mm) with `near`,
[`mps_matmul_align.mm`](mps_matmul_align.mm) with `align`, and
[`mps_gemm_trace.mm`](mps_gemm_trace.mm) with `trace`. All of them use only
Metal and MetalPerformanceShaders, plus the Objective-C runtime for `trace`.

`mps_matmul_2gib.mm` does the following:

1. It fills `A` (M×K) and `B` (K×N) with values in [0, 1].
2. It allocates result-sized shared buffers until it has one that crosses a
   2 GiB boundary and one that ends at least 16 MiB below the next boundary.
   The extra buffers are never touched, so they use address space but not
   memory.
3. It encodes the same `C = A * B` into both buffers, using one
   `MPSMatrixMultiplication` with no transposes, alpha 1, and beta 0.
4. It compares the two results and draws them side by side. Each map cell
   covers a block of rows and columns and shows `#` if the block contains a
   NaN. In a terminal, NaN cells and NaN values are shown in red.
5. It prints the top-left corner of the NaN band element by element.

`mps_matmul_near_2gib.mm` uses a placement heap (`MTLHeapTypePlacement`) to put
one result buffer at exact distances below a 2 GiB boundary: 16 MiB, 12 MiB,
8 MiB, and then every MiB down to 0. A final buffer starts at the boundary.
None of these buffers crosses the boundary. The program runs the same GEMM into
each buffer and counts the NaNs.

`mps_matmul_align.mm` allocates untouched 2 GiB buffers to walk the GPU address
space. For each alignment it finds, from 2^30 upward, it places the result
across an address that is a multiple of that power of two and of no higher
one, runs the GEMM, and compares the result with a reference computed away
from any 1 GiB boundary. It also prints the bits of the first NaN.

These three programs exit with 1 when they see the bug, 0 when they do not,
and 2 when they cannot place a buffer at a boundary.

`mps_gemm_trace.mm` replaces methods of the compute encoder class that Metal
returns on this machine, then encodes one `MPSMatrixMultiplication` per shape.
For each shape it prints the label of every pipeline MPS sets, which is the
kernel name, and the buffers, constants, threadgroup memory and dispatch sizes.

To build by hand:

```sh
clang++ -std=c++17 -O2 -fobjc-arc mps_matmul_2gib.mm \
    -framework Foundation -framework Metal -framework MetalPerformanceShaders \
    -o mps_matmul_2gib
```

Build the other programs the same way.

If the linker rejects the Command Line Tools SDK (`tapi error: malformed file`),
add `-isysroot $(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.
`run.sh` falls back to this SDK automatically.

## Observed output

On an Apple M2 (Mac14,2, 16 GB) with macOS 27.0.1 (26A434), the output is
identical on every run. After an update to macOS 27.2 (26B123), every program
in this repository prints the same output apart from GPU addresses.

```
device Apple M2, C = A(6000x64) * B(64x2500), result 60.0 MB
inside     gpuAddress [0x1500290000, 0x1503bc8700)  crosses 2 GiB: no   NaNs: 0  other mismatches: 0
straddling gpuAddress [0x157d650000, 0x1580f88700)  crosses 2 GiB: yes  NaNs: 1040000  other mismatches: 0
           NaN rows 3584..4415 (the boundary falls in row 4371)

                inside (correct)                         straddling ('#' = NaN)
rows    0- 199  |....................................|   |....................................|
   ...
rows 3200-3399  |....................................|   |....................................|
rows 3400-3599  |....................................|   |####################################|
rows 3600-3799  |....................................|   |####################################|
rows 3800-3999  |....................................|   |####################################|
rows 4000-4199  |....................................|   |####################################|
rows 4200-4399  |....................................|   |####################################| <- 2 GiB boundary
rows 4400-4599  |....................................|   |####################################|
rows 4600-4799  |....................................|   |....................................|
   ...
rows 5800-5999  |....................................|   |....................................|

straddling result, rows 3582-3586, columns 0-9:
row  3582   16.871  18.603  16.110  21.105  16.581  18.708  17.904  15.772  16.785  16.383
row  3583   16.080  17.100  15.257  18.190  15.616  17.668  16.973  14.730  15.678  16.003
row  3584      NaN  18.483     NaN  19.354     NaN  17.876     NaN  14.332     NaN  17.319
row  3585      NaN  17.045     NaN  16.846     NaN  17.543     NaN  14.533     NaN  14.810
row  3586      NaN  16.361     NaN  18.361     NaN  17.969     NaN  14.283     NaN  16.796

C[0,0]: CPU 15.402948, inside 15.402947
BUG REPRODUCED
```

With `4000 1000 32`, the result is 16 MB. Rows 1728 to 3839 hold 1,056,000
NaNs, and the boundary falls in row 3801.

Every element outside the NaN band matches the in-window result bit for bit.
The band is a block of whole rows around the boundary row. Within the band,
every even-numbered column (0, 2, 4, …) is NaN and every odd-numbered column is
correct. In the first run that is 832 rows × 1250 columns, or 1,040,000 NaNs;
in the second it is 2112 rows × 500 columns.

The band is not centered on the boundary. In the first run it starts 7.5 MiB
below the boundary and ends 0.4 MiB above it. `./run.sh near` shows the same
from the other side: a buffer that ends close below a boundary is corrupted
even though it does not cross it.

```
device Apple M2, C = A(6000x64) * B(64x2500), result 60.0 MB
each result buffer is placed alone; none crosses the 2 GiB boundary

gpuAddress [0x157b6c7900, 0x157f000000)  ends 16.000 MiB below the boundary  NaNs: 0
gpuAddress [0x157bac7900, 0x157f400000)  ends 12.000 MiB below the boundary  NaNs: 0
gpuAddress [0x157bec7900, 0x157f800000)  ends  8.000 MiB below the boundary  NaNs: 0
gpuAddress [0x157bfc7900, 0x157f900000)  ends  7.000 MiB below the boundary  NaNs: 60000
gpuAddress [0x157c0c7900, 0x157fa00000)  ends  6.000 MiB below the boundary  NaNs: 220000
gpuAddress [0x157c1c7900, 0x157fb00000)  ends  5.000 MiB below the boundary  NaNs: 380000
gpuAddress [0x157c2c7900, 0x157fc00000)  ends  4.000 MiB below the boundary  NaNs: 460000
gpuAddress [0x157c3c7900, 0x157fd00000)  ends  3.000 MiB below the boundary  NaNs: 620000
gpuAddress [0x157c4c7900, 0x157fe00000)  ends  2.000 MiB below the boundary  NaNs: 780000
gpuAddress [0x157c5c7900, 0x157ff00000)  ends  1.000 MiB below the boundary  NaNs: 860000
gpuAddress [0x157c6c7900, 0x1580000000)  ends  0.000 MiB below the boundary  NaNs: 1020000
gpuAddress [0x1580000000, 0x1583938700)  starts at the boundary              NaNs: 0

BUG REPRODUCED
```

## Which kernel is affected

`./run.sh trace` shows which kernel `MPSMatrixMultiplication` runs. We ran it,
and the reproductions, on these machines. The M4 is a bare-metal EC2
`mac-m4.metal` instance, not a virtual machine.

| Machine | macOS | MPS framework | Kernel for 6000 × 2500 × 64 | Affected? |
|---|---|---|---|---|
| Apple M2 (Mac14,2) | 27.0.1 (26A434) | 130.0.19 | not traced | yes |
| Apple M2 (Mac14,2) | 27.2 (26B123) | 130.1.4 | `MatrixMultiplyNNA14_M32_N128_singlebuffer_edgecase_all_types` | yes |
| Apple M4 (Mac16,10) | 27.0 (26A428) | 130.0.19 | `mmul_kernel_a16_float_float_float` | no |

The M4 has the same MPS framework version as the M2 on 27.0.1, but a newer GPU
family, and MPS gives it a different kernel. On the M4, `./run.sh` and
`./run.sh near` found no NaNs for 6000 × 2500 × 64 and 4000 × 1000 × 32, and
`./run.sh align` found none across multiples of 2^30 through 2^38 for
6000 × 2500 × 64. The M4 used
`mmul_kernel_a16_float_float_float` for every shape we traced, including
K = 7 and K = 8.

On the M2, MPS chooses among three kernels by shape:

| Shape (M × N × K) | Kernel | Affected? |
|---|---|---|
| K ≤ 7 (traced with K = 7) | `MatrixMultiplyNN` | no |
| M and N multiples of 64, K a multiple of 16 | `MatrixMultiplyNNA14_M32_N128_singlebuffer_all_types` | no |
| Any other shape | `MatrixMultiplyNNA14_M32_N128_singlebuffer_edgecase_all_types` | yes |

We derived the rule in the second row from 12 traced shapes. For example,
6016 × 2496 × 64 and 6016 × 2560 × 16 get the kernel without `edgecase`, while
6048 × 2560 × 64 (M a multiple of 32 only), 6016 × 2528 × 64 (N a multiple of 32 only)
and 6016 × 2560 × 24 (K a multiple of 8 only) get the `edgecase` kernel. Seven
shapes that follow the rule (6016 × 2560 × 64, 6016 × 2496 × 64,
6016 × 2560 × 16, 6016 × 2560 × 48, 4096 × 4096 × 64, 4096 × 4096 × 128 and
1024 × 4096 × 32) gave no NaNs in `./run.sh`, `./run.sh near` or `./run.sh align`.
6016 × 2560 × 24, where only K breaks the rule, gave NaNs in all three. So did
shapes where only M (6000 × 2560 × 64) or only N (6016 × 2500 × 64) breaks it.

Both A14 kernels run 16 × 16 threads per threadgroup. The grid sizes imply that
each threadgroup computes a 64 × 64 block of the result: 6000 × 2500 gives
40 × 94 threadgroups, and 4096 × 4096 gives 64 × 64. This matches the
corruption, which always covers whole multiples of 64 rows. In the 6000 × 2500 × 64 run, the NaN band starts at row
3584 = 56 × 64 and ends after row 4415 = 69 × 64 − 1.

`./run.sh align` also shows that the boundary can be any multiple of 2 GiB.
Results across multiples of 2^31 through 2^35 and 2^37 (the walk did not reach
2^36) all get the same NaN band, while a result across a multiple of 2^30 only
is correct. The NaNs have the bits `0x7fc00000`, the default NaN that
floating-point arithmetic produces.

## What does and does not trigger it

These results come from variants of the program above, on the M2:

| Configuration | Corrupted? |
|---|---|
| Result buffer crosses a 2 GiB-aligned address | yes |
| Result crosses a multiple of 4 GiB, 8 GiB, 16 GiB, 32 GiB or 128 GiB | yes |
| Result crosses, M and N multiples of 64, K a multiple of 16 | no |
| Result ends 7.5 MiB or less below a boundary, does not cross it | yes |
| Result ends 8 MiB or more below a boundary | no |
| Result starts at or above a boundary | no |
| Result matrix at an offset in a larger buffer that crosses a boundary, matrix ends 16 MiB below it | no |
| Result of 2 GiB or more (it always crosses one) | yes |
| Result crosses, interior dimension K ≥ 8 | yes |
| Result crosses, interior dimension K ≤ 7 | no |
| Left operand crosses a 2 GiB-aligned address, result does not | no |
| Result crosses only a 1 GiB-aligned address | no |
| Result crosses, buffer allocated from an `MTLHeap` | yes |
| Result crosses, private storage mode | yes |
| MPSGraph `matrixMultiplication` with the result crossing | no |
| `MPSNDArrayMatrixMultiplication` with the result crossing | no |
| `MPSMatrixVectorMultiplication` with the result crossing | no |
| Custom Metal compute kernels writing across the boundary | no |

What matters is the result matrix's own byte range, not the `MTLBuffer` that
holds it. How far below a boundary the corruption reaches depends a little on
the shape. For results of 4 MB to 256 MB with 500 to 20,000 rows, the first
NaNs appeared when the result ended 5.5 to 7.5 MiB below the boundary. A
96 × 156,250 result was corrupted only when it crossed the boundary.

Because a buffer of 2 GiB or more always spans such an address, any
`MPSMatrixMultiplication` with a result that large is affected, wherever it is
allocated. For example, a 60000×10000 float distance matrix (2.4 GB) is
affected.

PyTorch is not affected. PyTorch 2.9.1 does not use `MPSMatrixMultiplication`:
its MPS matmuls go through MPSGraph, and in some cases
`MPSNDArrayMatrixMultiplication`. `torch.mm`, `addmm`, `bmm` and `matmul`
writing into a tensor that crosses a boundary, and a 2.4 GB `A @ B`, all give
correct results.

## Workaround

The simplest option is to compute the GEMM with MPSGraph
`matrixMultiplication` or `MPSNDArrayMatrixMultiplication` instead. Neither
was affected in our tests. We have only tested this on the M2, so check
performance and results before switching.

To keep using `MPSMatrixMultiplication`, you can pad the problem so that M and
N are multiples of 64 and K is a multiple of 16. Pad `A` and `B` with zeros,
multiply into a padded result, and use its top-left M × N block. MPS then runs
the kernel without `edgecase`, which gave correct results in all our tests.
This relies on how MPS chooses kernels on the M2, which Apple does not
document and could change in any update, so check with `./run.sh trace` which
kernel your padded shapes get.

Otherwise, keep the results of `MPSMatrixMultiplication` away from 2 GiB
boundaries. Checking only whether the result crosses a boundary is not enough.
Before each GEMM:

1. Take the result's byte range, from `gpuAddress` plus the matrix offset to
   the end of its last row.
2. Extend the end of that range by a margin. We use 16 MiB, about twice the
   largest distance we measured. This value is not documented by Apple.
3. Check whether the extended range contains a 2 GiB-aligned address.

If it does:

1. Compute the product in row blocks (or groups of batch matrices) of at most
   256 MB.
2. Write each block into a scratch buffer that passes the same check.
3. Copy each block into place with a Metal kernel.

This bug was found in the [Juzhen](https://github.com/anewgithubname/Juzhen)
matrix library.

## Related reports

None of these reports describe this trigger, and both involve MPSGraph, which
this bug does not affect, so they are probably different bugs.

- [pytorch/pytorch#111634](https://github.com/pytorch/pytorch/issues/111634):
  wrong results from large matmuls on MPS. A maintainer described it as an
  MPS framework bug, and PyTorch worked around it by tiling (`[MPS] Fix
  torch.mm correctness for large matrices`).
- [Apple forums thread 705279](https://developer.apple.com/forums/thread/705279):
  MPSGraph matmul occasionally returns wrong results.
