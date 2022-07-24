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

## Documentation
Documentation can be generated and opened in browser with the following command:

```
cargo doc --no-deps --open
```

which will contain more information about using the Eidetic API.

## Examples
All examples can be found in the examples folder and can be run with the following format of command:

```
cargo run --release --example <example_name> --features="required-features"
```

If an example is run and it requires any features to be activated, it will notify you and you can simply enable those features.

## Website
The whole process of my learning deep learning by going through this book, the development of the initial version of the Eidetic API, and the refactor into this newer (slightly) more powerful version can be found documented in the articles on my website which can be found [HERE](https://forgottenmaster.github.io/posts/machinelearning/deeplearningfromscratch/). 