[![Check and Lint](https://github.com/ForgottenMaster/eidetic/actions/workflows/check-and-lint.yaml/badge.svg)](https://github.com/ForgottenMaster/eidetic/actions/workflows/check-and-lint.yaml)
[![Release Packaging](https://github.com/ForgottenMaster/eidetic/actions/workflows/release-packaging.yaml/badge.svg)](https://github.com/ForgottenMaster/eidetic/actions/workflows/release-packaging.yaml)
[![Test Coverage](https://github.com/ForgottenMaster/eidetic/actions/workflows/test-coverage.yaml/badge.svg)](https://github.com/ForgottenMaster/eidetic/actions/workflows/test-coverage.yaml)
[![codecov](https://codecov.io/gh/ForgottenMaster/eidetic/branch/main/graph/badge.svg?token=SNU0VO4WOU)](https://codecov.io/gh/ForgottenMaster/eidetic)

# Eidetic
A pure Rust library for creating, training, and using neural networks. Created as hands-on practice as I work my way through [Deep Learning From Scratch In Python](https://www.amazon.co.uk/Deep-Learning-Scratch-Building-Principles/dp/1492041416) and converting over to Rust as I go.

This API is implemented with the following pillars:
1. **Stability** - No dependency is included in the public API unless it has reached 1.0. This ensures that users have a stable public API to work with
2. **Embeddable** - The library doesn't use the standard library or a runtime so can be used in embedded environments
3. **Foolproof** - We will make copious use of typestates and error handling to ensure that the API cannot be misused in any way. Where possible correct API usage will be verified by the compiler
4. **Correctness** - We make use of unit testing and documentation testing to verify that the API is correct and functions as expected. Any example code in documentation will be correct and compile

# TODO
- [ ] Bias add
- [ ] Dense layer
- [ ] Dropout
- [ ] Mean squared error loss
- [ ] Softmax cross entropy loss
- [ ] Operation chaining
- [ ] SGD optimiser (with momentum)
- [ ] Learning rate linear decay
- [ ] Learning rate exponential decay
