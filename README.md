# EMCE - Exotic Monte Carlo Engine
Monte Carlo pricers for exotic derivatives in both Python / C++. The reason to code it up in two languages: 
* Both are languages commonly used in quantitative finance - implementing a solution in both languages provides a holistic way to learn the intracicies of both languages from bottom-up through comparison.
* The languages themselves boast different benefits - Python allows for a quicker development cycle where C++ provides the raw speed where there is demand for performance.

Contributions are *more* than welcome!

## C++
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/C++/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Clang-Format/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
* The library can be found [here](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/emce).
* Example tests can be found [here](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/emce/tests)
* Project to-dos are [here](https://github.com/users/KYLChiu/projects/2).

### Features
* Interface fully interporable between CUDA/C++, switching via enum.

### Requirements
* CUDA toolkit (tested on 11.7.0) with nvcc compiler.
* CMake (version >= 3.18, as we need CUDA with C++17 standard).
* C++17 compliant compiler.

### Install
Build:
```
cmake -S ./cpp/emce -B build
cmake --build
```
Run tests:
```
ctest --test-dir ./build
```

## Python
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Python/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)

[Exotic Monte Carlo Pricer](https://github.com/KYLChiu/the-big-learning-repo/tree/master/python/ExoticEngine): An equity/FX exotic Monte Carlo pricer.
* To do: see [this link](https://github.com/users/KYLChiu/projects/1) for a list of issues and project info.
* Examples: see unit tests [here](https://github.com/KYLChiu/the-big-learning-repo/tree/master/python/sandbox).

### Features (goals)
* advanced smile modelling (LV, SV, LSV)
* coherent multi-asset simulation (hybrid exotic engine)
* support wide range of path-dependent exotics
* advanced variance reduction techniques (control variant, Quasi random numbers)
* risk engine (FDM, AAD)

Python notebooks (analytics):
* [Vanilla put/call pricing with flat vol](https://github.com/KYLChiu/ExoticMonteCarloEngine/blob/fix-readme/python/ExoticEngine/VanilliaAnalytics.ipynb)
* Path dependent option pricing with flat vol (tbd)
* Barrier option risk profiles with flat vol (tbd)
* Simle dynamics under BS model with local vol (tbd)
* Smile dynamics under BS model with SV - Heston model (tbd)
* Convergence speed under different RNGs (tbd)
* Rainbow option pricing (tbd)
