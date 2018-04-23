# SaferCPlusPlus-BenchmarksGame

Apr 2018
  

This repository contains benchmarks originally from ["The Computer Language Benchmarks Game"](https://duckduckgo.com/?q=The+Computer+Language+Benchmarks+Game&ia=web), implemented in [SaferCPlusPlus](https://github.com/duneroadrunner/SaferCPlusPlus), a memory safe dialect/subset of C++. 

The orginal implementations using traditional (unsafe) C++ are in the subdirectories named "unchecked". The versions in the subdirectories named "strict" are made "SaferCPlusPlus compliant" by eliminating (to the extent possible) potentially (memory) unsafe C++ elements, generally substituting them with safe replacements provided by the SaferCPlusPlus library. The subdirectories named "checked" contain versions that are partially SaferCPlusPlus compliant, though at this point the "checked" implementations are somewhat out-of-date.

In addition to a sample measure of SaferCPlusPlus' performance cost, the implementations in this repository can be useful as code examples that help demonstrate how to use the SaferCPlusPlus library to add memory safety to your C++ code while preserving maximal performance.

The code has been tested with msvc2017 and g++7.2 (as of Apr 2018). (Note there appears to be a bug in g++5.4 that prevents it from compiling the converted "fasta" benchmark.)

Sample results:

benchmark | normalized elapsed time
--------- | -----------------------
n-body (args: "50000000") | 1.05
binary-trees (indices)* (args: "21") | 1.29
fasta (combined)** | 1.37
mandelbrot (args: "20000 out.pbm") | 1.25
spectral-norm (args: "7500") | 0.60***

##### platform: msvc2017/default optimizations/x64/Windows10/Haswell (Apr 2018):

Technically, the (geometric) mean result of these benchmarks is about 1.07. That is, the "SaferCPlusPlus compliant" implementations were, on average, about 7% slower than the original (unsafe) implementations. If we replace the the idiosyncratic, platform-specific "spectral-norm" result with the 1.02 result observed on our g++/UbuntuLinux platform, the mean becomes more like 1.19.

In any case, the precise results aren't very meaningful, as actual performance can vary significantly by platform. But if the question is whether C++ code can be made (largely) memory safe while maintaining performance in the ballpark of the original (unsafe) code, the answer appears to be yes.


&nbsp;


\* There is a marginally faster implementation of the "binary-trees" benchmark using pointers instead of indices. That implementation could also be translated to SaferCPlusPlus, but would be slower.  

\** On the msvc2017/Windows10 platform used, the fasta benchmark is actually dominated by writing to "standard out", even when that output is piped to null. This is not the case, for example, when tested on our g++/UbuntuLinux platform. So we added to the benchmark the option to disable writing to standard out. The results with and without this option were observed as follows:

benchmark | normalized elapsed time
--------- | -----------------------
fasta** (args: "5000000") (output piped to null) | 1.08
fasta** (args: "15000000") (output disabled in code) | 1.74

The "fasta (combined)" value used above is simply the (geometric) mean of these two results. For reference, we observed results of about 1.40 to 1.45 on our g++/UbuntuLinux platform, with the disabling of the output seemingly making little difference.

\*** The result for the "spectral-norm" benchmark might be a little surprising. While the details of the performance discrepancy have not yet been investigated, note that the original spectral-norm implementation uses OpenMP, while the converted version does not (for safety reasons). Perhaps support for OpenMP is suboptimal on the msvc2017/Windows10 platform used. For reference, we observed results of about 1.02 on our g++/UbuntuLinux platform.
