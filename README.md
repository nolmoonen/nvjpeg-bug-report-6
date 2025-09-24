# nvjpeg-bug-report-6

> [!CAUTION]
> Thanks to the nvJPEG team, this bug has been fixed in CUDA 13.0.

Demonstrates nvJPEG producing non-deterministic results for truncated images.

Tested with CUDA 12.8 driver version 570.86.15 on Ubuntu 22.04.1

The reproducer demonstrates this behavior with the below steps. Throughout the reproducer, only a single decode state is used.
1. Create a JPEG with random values (saved as `test_43.jpg`) and decode it.
2. Create a second JPEG with random values, different from the one in step 1. (saved as `test_44_0.jpg`). Truncate it (saved as `test_44_trunc_0.jpg`) and decode it. The result is stored as `test_44_trunc_decoded_0.ppm`.
3. Create a third JPEG with random values, different from the ones in step 1. or step 2. (saved as `test_45.jpg`) and decode it.
4. Recreate the exact image from step 2., and decode it again (saved as `test_44_trunc_decoded_1.ppm`).

The expected behavior is that `test_44_trunc_decoded_0.ppm` and `test_44_trunc_decoded_1.ppm` are the same, because the same truncated JPEG image is given as input. However, they are not the same, presumably because decoding the image in between sets some internal buffer that is not reset.

The images in this repository are the images produced by running `./repro 0.66`.
