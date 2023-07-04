# the-big-learning-repo
Learning stuff in cpp and python. Contributions are welcome!

## cpp
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Cpp-Ubuntu/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Clang-Format/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
[![codecov](https://codecov.io/gh/KYLChiu/the-big-learning-repo/branch/master/graph/badge.svg)](https://codecov.io/gh/KYLChiu/the-big-learning-repo)

### Memory:
* [KC] [**unique_ptr**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/memory/unique_ptr.hpp): example re-implementation of `std::unique_ptr<T>` with custom deleter.
  * To do: invoke empty base optimisation to ensure class size is same as `T*` if deleter is empty.
  * Examples: [here](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/sandbox/unique_ptr_test.cpp).

### Concurrency:
* [KC] [**future_chainer**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/concurrency/future_chainer.hpp): implements `std::future<T>` chaining by passing continuations to run (asychronously) on a successful future (returning a value of type `T`) or failed future (throwing an exception).
  * Examples: [here](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/sandbox/future_chainer_test.cpp).
* [KC] [**thread_pool**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/concurrency/thread_pool.hpp): a header only class allowing scheduling of work functions onto a pool of long-running worker threads. Synchronisation is fast and light-weight via C++20's `std::counting_semaphore`.
  * To do: implement work-stealing.
  * Examples: [here](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/sandbox/future_chainer_test.cpp)

### Option pricing:
* [KC] [**monte_carlo_engine**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/cuda/first_order_sde.cuh): a naive implementation of multi-threaded MC for first-order (deterministic coefficient) SDEs:
$$dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t$$
where $W_t$ is a Wiener process. Simulation of the SDE is done via Euler-Maruyama stepping.
  * To do: implement CUDA equivalent, add Milstein stepping, improve sampling.
  * Examples: [here](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/sandbox/cuda_test.cu).

## python
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Python/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)

### Option pricing:
* [JLo]: [Exotic Monte Carlo Pricer](https://github.com/KYLChiu/the-big-learning-repo/tree/master/python/ExoticEngine): An equity/FX exotic Monte Carlo pricer.
  * see [this link](https://github.com/users/KYLChiu/projects/1) for a list of issues and project info.
  * Examples: see unit tests [here](https://github.com/KYLChiu/the-big-learning-repo/tree/master/python/sandbox).
