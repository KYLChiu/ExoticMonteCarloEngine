# the-big-learning-repo
Learning stuff in cpp and python. Contributions are welcome!

## cpp
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Cpp-Ubuntu/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Clang-Format/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)
[![codecov](https://codecov.io/gh/KYLChiu/the-big-learning-repo/branch/master/graph/badge.svg)](https://codecov.io/gh/KYLChiu/the-big-learning-repo)

### Memory:
* [KC] [**unique_ptr**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/memory/unique_ptr.hpp): example re-implementation of `std::unique_ptr<T>` with custom deleter. 
  * To do: invoke empty base optimisation to ensure class size is same as `T*`.

### Concurrency:
* [KC] [**future_chainer**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/concurrency/future_chainer.hpp): implements future chaining by passing continuations to asychronously run on success (value) or failure (exception).
* [KC] [**thread_pool**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/concurrency/thread_pool.hpp): a light-weight class allowing scheduling of work functions onto a pool of long-running worker threads.
  * To do: implement work-stealing.

### Option pricing:
* [KC] [**monte_carlo_engine**](https://github.com/KYLChiu/the-big-learning-repo/blob/master/cpp/kc_utils/cuda/first_order_sde.cuh): a naive implementation of multi-threaded MC for first-order (deterministic coefficient) SDEs. Simulation is done via Euler-Maruyama. 
  * To do: implement CUDA equivalent, add Milstein stepping, improve sampling.

### Tests:
See [here](https://github.com/KYLChiu/the-big-learning-repo/tree/master/cpp/sandbox).

## python
[![Actions Status](https://github.com/KYLChiu/the-big-learning-repo/workflows/Python/badge.svg)](https://github.com/KYLChiu/the-big-learning-repo/actions)

### Option pricing:
* [JLo]: Exotic Monte Carlo Pricer
  * To do: implementation, with reference to [this book](https://www.amazon.co.uk/Patterns-Derivatives-Pricing-Mathematics-Finance/dp/0521721628).

