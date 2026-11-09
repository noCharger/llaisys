Running add.py... PASSED
Output for add.py:
Testing Ops.add on nvidia
   shape (2, 3) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.00721 ms (P50: 0.00715, P90: 0.00763, P99: 0.00836, Std: 0.00044)
        Torch Compile   time: 0.01988 ms (P50: 0.01931, P90: 0.02151, P99: 0.02838, Std: 0.00220)
        LLAISYS vs Compile speedup: 2.37x
        LLAISYS         time: 0.00837 ms (P50: 0.00834, P90: 0.00882, P99: 0.00954, Std: 0.00038)
        LLAISYS vs Eager speedup:   0.86x
   shape (2, 3) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.00679 ms (P50: 0.00691, P90: 0.00691, P99: 0.00716, Std: 0.00021)
        Torch Compile   time: 0.02282 ms (P50: 0.02217, P90: 0.02575, P99: 0.03485, Std: 0.00275)
        LLAISYS vs Compile speedup: 3.81x
        LLAISYS         time: 0.00599 ms (P50: 0.00596, P90: 0.00620, P99: 0.00644, Std: 0.00019)
        LLAISYS vs Eager speedup:   1.13x
   shape (2, 3) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.00708 ms (P50: 0.00715, P90: 0.00739, P99: 0.00740, Std: 0.00019)
        Torch Compile   time: 0.01956 ms (P50: 0.01931, P90: 0.02100, P99: 0.02577, Std: 0.00172)
        LLAISYS vs Compile speedup: 3.22x
        LLAISYS         time: 0.00607 ms (P50: 0.00596, P90: 0.00644, P99: 0.00644, Std: 0.00022)
        LLAISYS vs Eager speedup:   1.17x
   shape (512, 4096) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01265 ms (P50: 0.01264, P90: 0.01311, P99: 0.01335, Std: 0.00027)
        Torch Compile   time: 0.03130 ms (P50: 0.03088, P90: 0.03219, P99: 0.03891, Std: 0.00200)
        LLAISYS vs Compile speedup: 2.79x
        LLAISYS         time: 0.01121 ms (P50: 0.01121, P90: 0.01144, P99: 0.01526, Std: 0.00064)
        LLAISYS vs Eager speedup:   1.13x
   shape (512, 4096) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.00933 ms (P50: 0.00930, P90: 0.00954, P99: 0.00978, Std: 0.00021)
        Torch Compile   time: 0.02940 ms (P50: 0.02933, P90: 0.03104, P99: 0.03433, Std: 0.00160)
        LLAISYS vs Compile speedup: 3.34x
        LLAISYS         time: 0.00880 ms (P50: 0.00882, P90: 0.00906, P99: 0.01005, Std: 0.00051)
        LLAISYS vs Eager speedup:   1.06x
   shape (512, 4096) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.00932 ms (P50: 0.00930, P90: 0.00954, P99: 0.01002, Std: 0.00022)
        Torch Compile   time: 0.02789 ms (P50: 0.02766, P90: 0.02933, P99: 0.03410, Std: 0.00131)
        LLAISYS vs Compile speedup: 3.16x
        LLAISYS         time: 0.00881 ms (P50: 0.00882, P90: 0.00906, P99: 0.00938, Std: 0.00091)
        LLAISYS vs Eager speedup:   1.06x
Test passed!


Running argmax.py... PASSED
Output for argmax.py:
Testing Ops.argmax on nvidia
   shape (4,) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01694 ms (P50: 0.01657, P90: 0.01791, P99: 0.02071, Std: 0.00266)
        Torch Compile   time: 0.03924 ms (P50: 0.03850, P90: 0.04148, P99: 0.05083, Std: 0.00306)
        LLAISYS vs Compile speedup: 3.18x
        LLAISYS         time: 0.01232 ms (P50: 0.01192, P90: 0.01335, P99: 0.01506, Std: 0.00101)
        LLAISYS vs Eager speedup:   1.37x
   shape (4,) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01285 ms (P50: 0.01287, P90: 0.01338, P99: 0.01479, Std: 0.00059)
        Torch Compile   time: 0.04989 ms (P50: 0.04780, P90: 0.05684, P99: 0.06874, Std: 0.00698)
        LLAISYS vs Compile speedup: 5.25x
        LLAISYS         time: 0.00950 ms (P50: 0.00930, P90: 0.01001, P99: 0.01814, Std: 0.00147)
        LLAISYS vs Eager speedup:   1.35x
   shape (4,) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01328 ms (P50: 0.01335, P90: 0.01359, P99: 0.01408, Std: 0.00036)
        Torch Compile   time: 0.04943 ms (P50: 0.04840, P90: 0.05586, P99: 0.06371, Std: 0.00442)
        LLAISYS vs Compile speedup: 5.03x
        LLAISYS         time: 0.00982 ms (P50: 0.00978, P90: 0.01001, P99: 0.01083, Std: 0.00111)
        LLAISYS vs Eager speedup:   1.35x
   shape (4096,) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01492 ms (P50: 0.01478, P90: 0.01526, P99: 0.01612, Std: 0.00160)
        Torch Compile   time: 0.06263 ms (P50: 0.05913, P90: 0.07806, P99: 0.10302, Std: 0.01138)
        LLAISYS vs Compile speedup: 5.86x
        LLAISYS         time: 0.01070 ms (P50: 0.01049, P90: 0.01073, P99: 0.02269, Std: 0.00206)
        LLAISYS vs Eager speedup:   1.40x
   shape (4096,) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01441 ms (P50: 0.01407, P90: 0.01478, P99: 0.02003, Std: 0.00108)
        Torch Compile   time: 0.03704 ms (P50: 0.03648, P90: 0.04199, P99: 0.04849, Std: 0.00432)
        LLAISYS vs Compile speedup: 3.84x
        LLAISYS         time: 0.00964 ms (P50: 0.00954, P90: 0.00978, P99: 0.01503, Std: 0.00094)
        LLAISYS vs Eager speedup:   1.50x
   shape (4096,) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01296 ms (P50: 0.01264, P90: 0.01338, P99: 0.01669, Std: 0.00095)
        Torch Compile   time: 0.03890 ms (P50: 0.03791, P90: 0.04268, P99: 0.05866, Std: 0.00445)
        LLAISYS vs Compile speedup: 4.40x
        LLAISYS         time: 0.00883 ms (P50: 0.00882, P90: 0.00906, P99: 0.00931, Std: 0.00024)
        LLAISYS vs Eager speedup:   1.47x
Test passed!


Running embedding.py... PASSED
Output for embedding.py:
Testing Ops.embedding on nvidia
   idx_shape (1,) embd_shape (2, 3) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01382 ms (P50: 0.01383, P90: 0.01409, P99: 0.01552, Std: 0.00056)
        Torch Compile   time: 0.01974 ms (P50: 0.01931, P90: 0.02129, P99: 0.02529, Std: 0.00160)
        LLAISYS vs Compile speedup: 3.26x
        LLAISYS         time: 0.00606 ms (P50: 0.00596, P90: 0.00644, P99: 0.00674, Std: 0.00078)
        LLAISYS vs Eager speedup:   2.28x
   idx_shape (1,) embd_shape (2, 3) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01326 ms (P50: 0.01335, P90: 0.01359, P99: 0.01383, Std: 0.00022)
        Torch Compile   time: 0.01919 ms (P50: 0.01884, P90: 0.02029, P99: 0.02530, Std: 0.00152)
        LLAISYS vs Compile speedup: 3.21x
        LLAISYS         time: 0.00598 ms (P50: 0.00596, P90: 0.00620, P99: 0.00629, Std: 0.00093)
        LLAISYS vs Eager speedup:   2.22x
   idx_shape (1,) embd_shape (2, 3) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01348 ms (P50: 0.01335, P90: 0.01359, P99: 0.01439, Std: 0.00100)
        Torch Compile   time: 0.01971 ms (P50: 0.01955, P90: 0.02074, P99: 0.02456, Std: 0.00118)
        LLAISYS vs Compile speedup: 3.34x
        LLAISYS         time: 0.00590 ms (P50: 0.00596, P90: 0.00620, P99: 0.00644, Std: 0.00021)
        LLAISYS vs Eager speedup:   2.29x
   idx_shape (50,) embd_shape (512, 4096) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01379 ms (P50: 0.01383, P90: 0.01407, P99: 0.01431, Std: 0.00022)
        Torch Compile   time: 0.02681 ms (P50: 0.02646, P90: 0.02837, P99: 0.03410, Std: 0.00163)
        LLAISYS vs Compile speedup: 3.86x
        LLAISYS         time: 0.00695 ms (P50: 0.00691, P90: 0.00691, P99: 0.00983, Std: 0.00096)
        LLAISYS vs Eager speedup:   1.98x
   idx_shape (50,) embd_shape (512, 4096) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01356 ms (P50: 0.01335, P90: 0.01383, P99: 0.01417, Std: 0.00110)
        Torch Compile   time: 0.02663 ms (P50: 0.02599, P90: 0.02789, P99: 0.03272, Std: 0.00195)
        LLAISYS vs Compile speedup: 3.91x
        LLAISYS         time: 0.00681 ms (P50: 0.00691, P90: 0.00691, P99: 0.00715, Std: 0.00016)
        LLAISYS vs Eager speedup:   1.99x
   idx_shape (50,) embd_shape (512, 4096) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01364 ms (P50: 0.01359, P90: 0.01383, P99: 0.01696, Std: 0.00074)
        Torch Compile   time: 0.02636 ms (P50: 0.02599, P90: 0.02768, P99: 0.03226, Std: 0.00182)
        LLAISYS vs Compile speedup: 3.90x
        LLAISYS         time: 0.00676 ms (P50: 0.00668, P90: 0.00691, P99: 0.00692, Std: 0.00015)
        LLAISYS vs Eager speedup:   2.02x
Test passed!


Running linear.py... PASSED
Output for linear.py:
Testing Ops.linear on nvidia
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01130 ms (P50: 0.01121, P90: 0.01192, P99: 0.01314, Std: 0.00062)
        Torch Compile   time: 0.03139 ms (P50: 0.03088, P90: 0.03366, P99: 0.04032, Std: 0.00241)
        LLAISYS vs Compile speedup: 2.85x
        LLAISYS         time: 0.01100 ms (P50: 0.01073, P90: 0.01121, P99: 0.01220, Std: 0.00297)
        LLAISYS vs Eager speedup:   1.03x
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01110 ms (P50: 0.01121, P90: 0.01144, P99: 0.01145, Std: 0.00021)
        Torch Compile   time: 0.03250 ms (P50: 0.03195, P90: 0.03481, P99: 0.04125, Std: 0.00215)
        LLAISYS vs Compile speedup: 2.92x
        LLAISYS         time: 0.01112 ms (P50: 0.01097, P90: 0.01121, P99: 0.01563, Std: 0.00182)
        LLAISYS vs Eager speedup:   1.00x
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.01015 ms (P50: 0.01001, P90: 0.01025, P99: 0.01101, Std: 0.00055)
        Torch Compile   time: 0.03498 ms (P50: 0.03433, P90: 0.03676, P99: 0.04299, Std: 0.00234)
        LLAISYS vs Compile speedup: 3.57x
        LLAISYS         time: 0.00979 ms (P50: 0.00978, P90: 0.01001, P99: 0.01007, Std: 0.00060)
        LLAISYS vs Eager speedup:   1.04x
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.45913 ms (P50: 0.45919, P90: 0.46208, P99: 0.46375, Std: 0.00231)
        Torch Compile   time: 0.49752 ms (P50: 0.49782, P90: 0.49999, P99: 0.50188, Std: 0.00215)
        LLAISYS vs Compile speedup: 1.04x
        LLAISYS         time: 0.47967 ms (P50: 0.48280, P90: 0.48470, P99: 0.48525, Std: 0.00769)
        LLAISYS vs Eager speedup:   0.96x
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.15341 ms (P50: 0.15879, P90: 0.15974, P99: 0.16022, Std: 0.00651)
        Torch Compile   time: 0.15639 ms (P50: 0.15616, P90: 0.15786, P99: 0.16285, Std: 0.00166)
        LLAISYS vs Compile speedup: 1.08x
        LLAISYS         time: 0.14547 ms (P50: 0.14639, P90: 0.14734, P99: 0.14885, Std: 0.00284)
        LLAISYS vs Eager speedup:   1.05x
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.13488 ms (P50: 0.13447, P90: 0.13542, P99: 0.14554, Std: 0.00267)
        Torch Compile   time: 0.15958 ms (P50: 0.15926, P90: 0.16093, P99: 0.16413, Std: 0.00190)
        LLAISYS vs Compile speedup: 1.17x
        LLAISYS         time: 0.13642 ms (P50: 0.13614, P90: 0.13709, P99: 0.13885, Std: 0.00128)
        LLAISYS vs Eager speedup:   0.99x
Test passed!


Running rms_norm.py... PASSED
Output for rms_norm.py:
Testing Ops.rms_norm on nvidia
   shape (1, 4) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.02858 ms (P50: 0.02813, P90: 0.02980, P99: 0.03986, Std: 0.00222)
        Torch Compile   time: 0.02056 ms (P50: 0.02003, P90: 0.02244, P99: 0.02720, Std: 0.00184)
        LLAISYS vs Compile speedup: 3.07x
        LLAISYS         time: 0.00670 ms (P50: 0.00668, P90: 0.00691, P99: 0.00697, Std: 0.00060)
        LLAISYS vs Eager speedup:   4.26x
   shape (1, 4) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.02586 ms (P50: 0.02551, P90: 0.02670, P99: 0.03103, Std: 0.00116)
        Torch Compile   time: 0.01986 ms (P50: 0.01967, P90: 0.02151, P99: 0.02672, Std: 0.00180)
        LLAISYS vs Compile speedup: 3.03x
        LLAISYS         time: 0.00656 ms (P50: 0.00644, P90: 0.00668, P99: 0.01002, Std: 0.00060)
        LLAISYS vs Eager speedup:   3.94x
   shape (1, 4) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.02589 ms (P50: 0.02575, P90: 0.02623, P99: 0.03506, Std: 0.00152)
        Torch Compile   time: 0.02005 ms (P50: 0.01979, P90: 0.02146, P99: 0.02410, Std: 0.00135)
        LLAISYS vs Compile speedup: 3.09x
        LLAISYS         time: 0.00649 ms (P50: 0.00644, P90: 0.00668, P99: 0.00691, Std: 0.00019)
        LLAISYS vs Eager speedup:   3.99x
   shape (512, 4096) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.03602 ms (P50: 0.03552, P90: 0.03719, P99: 0.04118, Std: 0.00247)
        Torch Compile   time: 0.03293 ms (P50: 0.03266, P90: 0.03433, P99: 0.03604, Std: 0.00123)
        LLAISYS vs Compile speedup: 2.28x
        LLAISYS         time: 0.01443 ms (P50: 0.01431, P90: 0.01478, P99: 0.01529, Std: 0.00049)
        LLAISYS vs Eager speedup:   2.50x
   shape (512, 4096) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.03402 ms (P50: 0.03362, P90: 0.03457, P99: 0.04173, Std: 0.00130)
        Torch Compile   time: 0.03074 ms (P50: 0.03040, P90: 0.03266, P99: 0.03841, Std: 0.00185)
        LLAISYS vs Compile speedup: 2.53x
        LLAISYS         time: 0.01213 ms (P50: 0.01192, P90: 0.01240, P99: 0.01342, Std: 0.00087)
        LLAISYS vs Eager speedup:   2.80x
   shape (512, 4096) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.03376 ms (P50: 0.03362, P90: 0.03433, P99: 0.04104, Std: 0.00143)
        Torch Compile   time: 0.03111 ms (P50: 0.03076, P90: 0.03221, P99: 0.03652, Std: 0.00171)
        LLAISYS vs Compile speedup: 2.60x
        LLAISYS         time: 0.01195 ms (P50: 0.01192, P90: 0.01216, P99: 0.01240, Std: 0.00020)
        LLAISYS vs Eager speedup:   2.83x
Test passed!


Running rope.py... PASSED
Output for rope.py:
Testing Ops.rope on nvidia
   shape (2, 1, 4) range (0, 2) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.09785 ms (P50: 0.09799, P90: 0.10231, P99: 0.10848, Std: 0.00381)
        Torch Compile   time: 0.02112 ms (P50: 0.02050, P90: 0.02267, P99: 0.02981, Std: 0.00200)
        LLAISYS vs Compile speedup: 3.21x
        LLAISYS         time: 0.00659 ms (P50: 0.00644, P90: 0.00668, P99: 0.00699, Std: 0.00082)
        LLAISYS vs Eager speedup:   14.85x
   shape (2, 1, 4) range (0, 2) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.08708 ms (P50: 0.08655, P90: 0.08776, P99: 0.10109, Std: 0.00263)
        Torch Compile   time: 0.02130 ms (P50: 0.02098, P90: 0.02291, P99: 0.02960, Std: 0.00190)
        LLAISYS vs Compile speedup: 3.38x
        LLAISYS         time: 0.00630 ms (P50: 0.00620, P90: 0.00644, P99: 0.00668, Std: 0.00018)
        LLAISYS vs Eager speedup:   13.82x
   shape (2, 1, 4) range (0, 2) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.08714 ms (P50: 0.08655, P90: 0.08845, P99: 0.09611, Std: 0.00226)
        Torch Compile   time: 0.02142 ms (P50: 0.02098, P90: 0.02289, P99: 0.02959, Std: 0.00209)
        LLAISYS vs Compile speedup: 3.39x
        LLAISYS         time: 0.00632 ms (P50: 0.00644, P90: 0.00644, P99: 0.00668, Std: 0.00020)
        LLAISYS vs Eager speedup:   13.78x
   shape (512, 4, 4096) range (512, 1024) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.27717 ms (P50: 0.27704, P90: 0.27919, P99: 0.28423, Std: 0.00198)
        Torch Compile   time: 0.12827 ms (P50: 0.12779, P90: 0.13113, P99: 0.13733, Std: 0.00214)
        LLAISYS vs Compile speedup: 3.30x
        LLAISYS         time: 0.03886 ms (P50: 0.03886, P90: 0.03910, P99: 0.03941, Std: 0.00083)
        LLAISYS vs Eager speedup:   7.13x
   shape (512, 4, 4096) range (512, 1024) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.22098 ms (P50: 0.21780, P90: 0.23117, P99: 0.23583, Std: 0.00604)
        Torch Compile   time: 0.13064 ms (P50: 0.12922, P90: 0.13685, P99: 0.14020, Std: 0.00348)
        LLAISYS vs Compile speedup: 3.33x
        LLAISYS         time: 0.03920 ms (P50: 0.03910, P90: 0.03958, P99: 0.04005, Std: 0.00034)
        LLAISYS vs Eager speedup:   5.64x
   shape (512, 4, 4096) range (512, 1024) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.20230 ms (P50: 0.19789, P90: 0.20006, P99: 0.22780, Std: 0.03702)
        Torch Compile   time: 0.13123 ms (P50: 0.13173, P90: 0.13399, P99: 0.13733, Std: 0.00259)
        LLAISYS vs Compile speedup: 3.35x
        LLAISYS         time: 0.03920 ms (P50: 0.03910, P90: 0.03958, P99: 0.03992, Std: 0.00118)
        LLAISYS vs Eager speedup:   5.16x
Test passed!


Running self_attention.py... PASSED
Output for self_attention.py:
Testing Ops.self_attention on nvidia
   qlen=2 kvlen=2 nh=1 nkvh=1 hd=4 dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.08859 ms (P50: 0.09024, P90: 0.09263, P99: 0.10514, Std: 0.00511)
        Torch Compile   time: 0.05983 ms (P50: 0.05913, P90: 0.06344, P99: 0.06701, Std: 0.00226)
        LLAISYS vs Compile speedup: 4.40x
        LLAISYS         time: 0.01360 ms (P50: 0.01359, P90: 0.01383, P99: 0.01407, Std: 0.00021)
        LLAISYS vs Eager speedup:   6.52x
   qlen=2 kvlen=2 nh=1 nkvh=1 hd=4 dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.08400 ms (P50: 0.08368, P90: 0.08516, P99: 0.09156, Std: 0.00177)
        Torch Compile   time: 0.06953 ms (P50: 0.06890, P90: 0.07205, P99: 0.07869, Std: 0.00230)
        LLAISYS vs Compile speedup: 4.68x
        LLAISYS         time: 0.01486 ms (P50: 0.01454, P90: 0.01502, P99: 0.01961, Std: 0.00120)
        LLAISYS vs Eager speedup:   5.65x
   qlen=2 kvlen=2 nh=1 nkvh=1 hd=4 dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.08463 ms (P50: 0.08404, P90: 0.08609, P99: 0.09371, Std: 0.00204)
        Torch Compile   time: 0.06954 ms (P50: 0.06890, P90: 0.07200, P99: 0.07756, Std: 0.00248)
        LLAISYS vs Compile speedup: 4.65x
        LLAISYS         time: 0.01495 ms (P50: 0.01502, P90: 0.01526, P99: 0.01550, Std: 0.00027)
        LLAISYS vs Eager speedup:   5.66x
   qlen=5 kvlen=11 nh=4 nkvh=2 hd=8 dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.08484 ms (P50: 0.08345, P90: 0.09181, P99: 0.09543, Std: 0.00372)
        Torch Compile   time: 0.13355 ms (P50: 0.13459, P90: 0.13812, P99: 0.14211, Std: 0.00433)
        LLAISYS vs Compile speedup: 6.23x
        LLAISYS         time: 0.02145 ms (P50: 0.02122, P90: 0.02170, P99: 0.02204, Std: 0.00110)
        LLAISYS vs Eager speedup:   3.96x
   qlen=5 kvlen=11 nh=4 nkvh=2 hd=8 dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.09534 ms (P50: 0.09322, P90: 0.09944, P99: 0.11232, Std: 0.00417)
        Torch Compile   time: 0.11483 ms (P50: 0.11003, P90: 0.14024, P99: 0.14760, Std: 0.01124)
        LLAISYS vs Compile speedup: 4.70x
        LLAISYS         time: 0.02443 ms (P50: 0.02432, P90: 0.02480, P99: 0.02675, Std: 0.00076)
        LLAISYS vs Eager speedup:   3.90x
   qlen=5 kvlen=11 nh=4 nkvh=2 hd=8 dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.09449 ms (P50: 0.09131, P90: 0.10753, P99: 0.11875, Std: 0.01148)
        Torch Compile   time: 0.11170 ms (P50: 0.11134, P90: 0.11470, P99: 0.11945, Std: 0.00224)
        LLAISYS vs Compile speedup: 4.85x
        LLAISYS         time: 0.02302 ms (P50: 0.02289, P90: 0.02337, P99: 0.02372, Std: 0.00123)
        LLAISYS vs Eager speedup:   4.10x
Test passed!


Running swiglu.py... PASSED
Output for swiglu.py:
Testing Ops.swiglu on nvidia
   shape (2, 3) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.03810 ms (P50: 0.03767, P90: 0.03910, P99: 0.05058, Std: 0.00249)
        Torch Compile   time: 0.03658 ms (P50: 0.03576, P90: 0.04029, P99: 0.04606, Std: 0.00326)
        LLAISYS vs Compile speedup: 4.86x
        LLAISYS         time: 0.00752 ms (P50: 0.00739, P90: 0.00787, P99: 0.00811, Std: 0.00021)
        LLAISYS vs Eager speedup:   5.06x
   shape (2, 3) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.05154 ms (P50: 0.05102, P90: 0.05245, P99: 0.05940, Std: 0.00191)
        Torch Compile   time: 0.03659 ms (P50: 0.03600, P90: 0.03915, P99: 0.04890, Std: 0.00327)
        LLAISYS vs Compile speedup: 5.06x
        LLAISYS         time: 0.00723 ms (P50: 0.00715, P90: 0.00739, P99: 0.00763, Std: 0.00020)
        LLAISYS vs Eager speedup:   7.13x
   shape (2, 3) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.05109 ms (P50: 0.05054, P90: 0.05157, P99: 0.05961, Std: 0.00165)
        Torch Compile   time: 0.03772 ms (P50: 0.03695, P90: 0.04056, P99: 0.05064, Std: 0.00340)
        LLAISYS vs Compile speedup: 5.16x
        LLAISYS         time: 0.00731 ms (P50: 0.00715, P90: 0.00739, P99: 0.00790, Std: 0.00043)
        LLAISYS vs Eager speedup:   6.99x
   shape (512, 4096) dtype <f32>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.04483 ms (P50: 0.04458, P90: 0.04530, P99: 0.05104, Std: 0.00121)
        Torch Compile   time: 0.04735 ms (P50: 0.04721, P90: 0.04985, P99: 0.05675, Std: 0.00235)
        LLAISYS vs Compile speedup: 3.40x
        LLAISYS         time: 0.01392 ms (P50: 0.01383, P90: 0.01407, P99: 0.01431, Std: 0.00017)
        LLAISYS vs Eager speedup:   3.22x
   shape (512, 4096) dtype <f16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.05497 ms (P50: 0.05460, P90: 0.05531, P99: 0.06247, Std: 0.00153)
        Torch Compile   time: 0.04720 ms (P50: 0.04649, P90: 0.05057, P99: 0.05568, Std: 0.00324)
        LLAISYS vs Compile speedup: 3.70x
        LLAISYS         time: 0.01276 ms (P50: 0.01264, P90: 0.01311, P99: 0.01337, Std: 0.00036)
        LLAISYS vs Eager speedup:   4.31x
   shape (512, 4096) dtype <bf16>
    Benchmarking Torch Eager...
    Benchmarking LLAISYS...
    Benchmarking Torch Compile...
        Torch Eager     time: 0.05452 ms (P50: 0.05412, P90: 0.05484, P99: 0.06271, Std: 0.00160)
        Torch Compile   time: 0.04614 ms (P50: 0.04554, P90: 0.04888, P99: 0.05627, Std: 0.00241)
        LLAISYS vs Compile speedup: 3.61x
        LLAISYS         time: 0.01278 ms (P50: 0.01264, P90: 0.01290, P99: 0.01338, Std: 0.00045)
        LLAISYS vs Eager speedup:   4.27x
Test passed!



========================================
Summary: 8 PASSED, 0 FAILED
========================================
All operator tests passed!
