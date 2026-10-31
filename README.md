# MPSMatrixMultiplication writes NaNs into results near a 2 GiB GPU address

`MPSMatrixMultiplication` fills part of its result with NaNs when the result
spans a 2 GiB-aligned GPU virtual address, that is, when `gpuAddress / 2^31`
differs between the first and last byte of the result. It also does so when
the result does not cross such an address but ends less than about 8 MiB
below one. The same multiplication into an identical buffer away from a
boundary is correct. The input values and the result size do not matter: a
4 MB result is enough. The interior dimension K does matter: K of 7 or less is
not affected, and K of 8 or more is.

We found no mention of this in Apple's documentation or developer forums.

## Test on your Mac

With the Command Line Tools installed (`xcode-select --install`), run:

```sh
git clone --depth 1 https://github.com/anewgithubname/mps-matmul-2gib-bug && cd mps-matmul-2gib-bug && ./run.sh; ./run.sh near; sw_vers; sysctl -n hw.model
```

It takes a few seconds. Each of the two programs ends with `BUG REPRODUCED` or
`no corruption observed`. Either way, please
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
```

[`run.sh`](run.sh) builds [`mps_matmul_2gib.mm`](mps_matmul_2gib.mm), or
[`mps_matmul_near_2gib.mm`](mps_matmul_near_2gib.mm) with `near`, into
`$TMPDIR` and runs it. Both programs use only Metal and
MetalPerformanceShaders.

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

Both programs exit with 1 when they see the bug, 0 when they do not, and 2
when they cannot place a buffer at a boundary.

To build by hand:

```sh
clang++ -std=c++17 -O2 -fobjc-arc mps_matmul_2gib.mm \
    -framework Foundation -framework Metal -framework MetalPerformanceShaders \
    -o mps_matmul_2gib
```

Build `mps_matmul_near_2gib.mm` the same way.

If the linker rejects the Command Line Tools SDK (`tapi error: malformed file`),
add `-isysroot $(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.
`run.sh` falls back to this SDK automatically.

## Observed output

On an Apple M2 (Mac14,2, 16 GB) with macOS 27.0.1 (26A434), the output is
identical on every run:

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

## What does and does not trigger it

These results come from variants of the program above, on the same machine:

| Configuration | Corrupted? |
|---|---|
| Result buffer crosses a 2 GiB-aligned address | yes |
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
was affected in our tests. We have only tested this on the machine above, so
check performance and results before switching.

To keep using `MPSMatrixMultiplication`, keep its results away from 2 GiB
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
