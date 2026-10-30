# MPSMatrixMultiplication writes NaNs when the result straddles a 2 GiB GPU address

`MPSMatrixMultiplication` fills part of its result with NaNs when the result
`MTLBuffer` spans a 2 GiB-aligned GPU virtual address, that is, when
`gpuAddress / 2^31` differs between the first and last byte of the buffer. The
same multiplication into an identical buffer that does not cross a boundary is
correct. The input values and the result size do not matter: a 16 MB result
is enough. The interior dimension K does matter: K of 7 or less is not
affected, and K of 8 or more is.

We found no mention of this in Apple's documentation or developer forums.

## Test on your Mac

With the Command Line Tools installed (`xcode-select --install`), run:

```sh
git clone --depth 1 https://github.com/anewgithubname/mps-matmul-2gib-bug && cd mps-matmul-2gib-bug && ./run.sh; sw_vers; sysctl -n hw.model
```

It takes a few seconds and ends with `BUG REPRODUCED` or `no corruption
observed`. Either way, please
[open an issue](https://github.com/anewgithubname/mps-matmul-2gib-bug/issues/new)
with the full output, so we can track which chips and macOS versions are
affected. Results from a virtual machine do not count: its GPU is
paravirtualized.

## Reproduce

```sh
./run.sh            # M N K default to 6000 2500 64
./run.sh 4000 1000 32
```

[`run.sh`](run.sh) builds [`mps_matmul_2gib.mm`](mps_matmul_2gib.mm) into
`$TMPDIR` and runs it. The program uses only Metal and
MetalPerformanceShaders. It does the following:

1. It fills `A` (M×K) and `B` (K×N) with values in [0, 1].
2. It allocates result-sized shared buffers until it has one that lies inside a
   2 GiB-aligned window and one that crosses a 2 GiB boundary. The extra
   buffers are never touched, so they use address space but not memory.
3. It encodes the same `C = A * B` into both buffers, using one
   `MPSMatrixMultiplication` with no transposes, alpha 1, and beta 0.
4. It compares the two results and draws them side by side. Each map cell
   covers a block of rows and columns and shows `#` if the block contains a
   NaN. In a terminal, NaN cells and NaN values are shown in red.
5. It prints the top-left corner of the NaN band element by element.

The program exits with 1 when it sees the bug, 0 when the results agree, and 2
when it cannot place a buffer across a boundary.

To build by hand:

```sh
clang++ -std=c++17 -O2 -fobjc-arc mps_matmul_2gib.mm \
    -framework Foundation -framework Metal -framework MetalPerformanceShaders \
    -o mps_matmul_2gib
```

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

## What does and does not trigger it

These results come from variants of the program above, on the same machine:

| Configuration | Corrupted? |
|---|---|
| Result buffer crosses a 2 GiB-aligned address | yes |
| Result of 2 GiB or more (it always crosses one) | yes |
| Result crosses, interior dimension K ≥ 8 | yes |
| Result crosses, interior dimension K ≤ 7 | no |
| Left operand crosses a 2 GiB-aligned address, result does not | no |
| Result crosses only a 1 GiB-aligned address | no |
| Same GEMM, result inside a 2 GiB window | no |
| `MPSMatrixVectorMultiplication` with the result crossing | no |
| Custom Metal compute kernels writing across the boundary | no |

Because a buffer of 2 GiB or more always spans such an address, any MPS GEMM
with a result that large is affected, wherever it is allocated. For example,
a 60000×10000 float distance matrix (2.4 GB) is affected.

## Workaround

Keep MPS GEMM results out of buffers that cross a 2 GiB boundary. Before each
GEMM, check the result buffer's `gpuAddress` range. If it crosses a boundary:

1. Compute the product in row blocks (or groups of batch matrices) of at most
   256 MB.
2. Write each block into a scratch buffer that does not cross a boundary.
3. Copy each block into place with a Metal kernel.

The [Juzhen](https://github.com/anewgithubname/Juzhen) matrix library, where
this bug was found, does this in `mpsGemm` and `mpsGemmBatched`
(`cpp/metal/MPSWrapper.mm`). Its test `tests/testMPSGemmAddressBoundary.cu`
places results across the boundary and fails with NaNs when the workaround is
disabled.

## Related reports

None of these reports describe this trigger. The closest is the PyTorch issue.

- [pytorch/pytorch#111634](https://github.com/pytorch/pytorch/issues/111634):
  wrong results from large matmuls on MPS. A maintainer described it as an
  MPS framework bug, and PyTorch worked around it by tiling (`[MPS] Fix
  torch.mm correctness for large matrices`).
- [Apple forums thread 705279](https://developer.apple.com/forums/thread/705279):
  MPSGraph matmul occasionally returns wrong results.
