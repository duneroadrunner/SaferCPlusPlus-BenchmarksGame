# SaferCPlusPlus-BenchmarksGame

Oct 2016
  

This repository contains benchmarks originally from ["The Computer Language Benchmarks Game"](http://benchmarksgame.alioth.debian.org), implemented in [SaferCPlusPlus](https://github.com/duneroadrunner/SaferCPlusPlus), a memory safe dialect/subset of C++. Three types of implementations are provided - "unchecked", "checked" and "strict". The "unchecked" versions are the original implementations using traditional (unsafe) C++. The "checked" and "strict" versions use the SaferCPlusPlus library to increase memory safety. The "checked" versions are technically not quite as safe as the "strict" versions, the main differences being:

- "Checked" implementations accomodate the use of C++ references, even though they are technically unsafe.
- "Checked" implementations accomodate the use of mse::msearray<> and mse::msevector<> in lieu of their slightly safer counterparts, mse::mstd::array<> and mse::mstd::vector<>.
- "Strict" implementations generally define the MSE_MSEARRAY_USE_MSE_PRIMITIVES and MSE_MSEVECTOR_USE_MSE_PRIMITIVES preprocessor directives which cause the arrays and vectors to use mse::CInt and mse::CSize_t in their implementation and interfaces in lieu of their "less safe" native counterparts.

Note that these are fairly direct translations of the original C++ implementation. This often results in a lot of redundant bounds checking. In many cases the code could be reworked to significantly reduce the redundancy.  

The provided code has been tested to work with msvc2015 and g++5.3 (as of Oct 2016).  

Sample results (normalized elapsed time):

#### n-body (args: "50000000")

unchecked | checked | strict
--------- | ------- | ------
1.00 | 1.08 | 1.15

#### binary-trees (indices)* (args: "20")

unchecked | checked | strict
--------- | ------- | ------
1.00 | 1.10 | 1.44

#### fasta** (args: "5000000") (standard output piped to null)
unchecked | checked | strict
--------- | ------- | ------
1.00 | 1.77 | 1.86

#### mandelbrot (args: "10000 out.pbm")

unchecked | checked | strict
--------- | ------- | ------
1.00 | 1.17 | 1.37

#### spectral-norm (args: "5500")

unchecked | checked | strict
--------- | ------- | ------
1.00 | 1.00 | 1.00

##### platform: msvc2015/default optimizations/x64/Windows10/Haswell (Oct 2016):

#### (geometric) mean result

unchecked | checked | strict
--------- | ------- | ------
1.00 | 1.20 | 1.33

\* There is a marginally faster implementation of binary-trees using pointers instead of indices. That implementation could also be translated to SaferCPlusPlus, but would be slower.  

\** The performance discrepencies in the "fasta" benchmark are largely not related to the performance of the SaferCPlusPlus elements, but rather the "safer" programming style SaferCPlusPlus encourages with respect to sharing objects between asynchronous threads. In particular, the original "unchecked" implementation of the fasta benchmark engages in a lot of direct writes to global and static variables from multiple different asynchronous threads. As a general rule, SaferCPlusPlus considers this an unsafe practice. And so the "checked" and "strict" SaferCPlusPlus implementations instead replace those direct accesses with safer ones, not bothering to factor the overhead out of the inner loops.


