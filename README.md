# EMCE - Exotic Monte Carlo Engine
Monte Carlo pricers for exotic derivaities in both Python / C++. The reason to code it up in two languages: 
* Both are languages commonly used in quantitative finance - implementing a solution in both languages provides a holistic way to learn the intracicies of both languages from bottom-up.
* The languages themselves boast different benefits - Python (usually) provides a quicker and smoother development cycle where C++ provides the raw speed where there is demand for performance.

Contributions are *more* than welcome!

## cpp
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/C++/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Clang-Format/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
[![codecov](https://codecov.io/gh/KYLChiu/the-big-learning-repo/branch/master/graph/badge.svg)](https://codecov.io/gh/KYLChiu/the-big-learning-repo)

* [KC] (https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/monte_carlo_pricer/monte_carlo_pricer.cuh): a (header-only) Monte Carlo pricer implemented in CUDA/multi-threaded C++.
  * To do: see [this link](https://github.com/users/KYLChiu/projects/2)
  * Examples: see unit tests [here](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/sandbox/mc_pricer_test.cu).

## python
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Python/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
* [JLo]: [Exotic Monte Carlo Pricer](https://github.com/KYLChiu/the-big-learning-repo/tree/master/python/ExoticEngine): An equity/FX exotic Monte Carlo pricer.
  * To do: see [this link](https://github.com/users/KYLChiu/projects/1) for a list of issues and project info.
  * Examples: see unit tests [here](https://github.com/KYLChiu/the-big-learning-repo/tree/master/python/sandbox).

