<h1 align="center">
    <a href="http://atomashpolskiy.github.io/rustface/">Rustface</a>
</h1>

<p align="center"><strong>
<sup>
<br/>SeetaFace detection library for the Rust programming language
</sup>
</strong></p>

<p align="center">
    <img src="https://atomashpolskiy.github.io/static/img/scientists.png" alt="Bt Example">
    <i>Example of demo program output</i>
</p>

<p align="left">
    <a href="https://crates.io/crates/rustface">
        <img src="https://img.shields.io/crates/v/rustface.svg"
             alt="crates.io">
    </a>
    <a href="https://docs.rs/rustface">
            <img src="https://docs.rs/rustface/badge.svg"
                 alt="docs.rs">
        </a>
    <a href="https://travis-ci.org/atomashpolskiy/rustface">
        <img src="https://img.shields.io/travis/atomashpolskiy/rustface/master.svg"
             alt="Linux build">
    </a>    
    <a href="https://opensource.org/licenses/BSD-2-Clause">
        <img src="https://img.shields.io/badge/license-BSD-blue.svg"
             alt="License">
    </a>
</p>

* **[SEETAFACE C++](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceDetection)** – Github repository for the original library
* **[PYTHON BINDINGS](https://github.com/torchbox/rustface-py)** – call Rustface from Python code
* **[LICENSE](https://github.com/atomashpolskiy/rustface/blob/master/LICENSE)** – licensed under permissive BSD 2-Clause

## About

SeetaFace Detection is an implementation of Funnel-Structured cascade, which is designed for **real-time** multi-view face detection. FuSt aims at a good trade-off between accuracy and speed by using a coarse-to-fine structure. It consists of multiple view-specific fast LAB cascade classifiers at early stages, followed by coarse Multilayer Perceptron (MLP) cascades at later stages. The final stage is one unified fine MLP cascade, processing all proposed windows in a centralized style. 

[Read more...](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceDetection#seetaface-detection)

## Performance

Crude manual benchmarking shows that the Rust version is _slightly faster_ than the original C++ version. Here are some numbers for a [medium-sized image with 29 persons](https://github.com/atomashpolskiy/rustface/tree/master/assets/test/scientists.jpg), which you may see above in this readme:

```
Image size: 1666x1136
Number of faces: 29

CPU: 2,3 GHz Intel Core i7
Single-threaded (OpenMP disabled, Rayon threads set to 1)
SIMD enabled
LTO disabled

* Original *
samples (ms): 893,893,891,883,884,883,890,908,893,879
mean (ms): 889.7
stddev (ms): 7.785

* Rustface *
samples (ms): 867,861,851,850,856,847,855,851,850,861
mean (ms): 854.9
stddev (ms): 6.024
```

In this particular test the Rust version has been **4% faster on average** than its C++ counterpart.

When using multiple threads and enabling LTO (link-time optimization), Rust performance is a tad better (I observe a **8% boost**):

```
Multi-threaded (Rayon threads set to 2)
LTO enabled

* Rustface *
samples (ms): 787,789,795,795,787,785,791,799,795,788
mean (ms): 791.1
stddev (ms): 4.39
```

## Usage example

```rust
extern crate rustface;

use rustface::{Detector, FaceInfo, ImageData};

fn main() {
    let mut detector = rustface::create_detector("/path/to/model").unwrap();
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);
    
    let mut image = ImageData::new(bytes, width, height);
    for face in detector.detect(&mut image).into_iter() {
        // print confidence score and coordinates
        println!("found face: {:?}", face);
    }
}
```

## How to build

The project is a library crate and also contains a runnable example for demonstration purposes.

Due to usage of [experimental stdsimd crate](https://github.com/rust-lang-nursery/stdsimd) for SIMD support, the project relies on the nightly Rust toolchain, so you'll need to install it and set it as the default:

```
rustup default nightly
```

Then just use the standard Cargo `build` command:

```
cargo build --release
```

## Run demo

Code for the demo is located in `examples/image_demo.rs` file. It performs face detection for a given image and saves the result into a file in the working directory.

The simplest way to run the demo is to use the `bin/test.sh` script:

```
./bin/test.sh <path-to-image>
```

Please note that this library makes use of [Rayon](https://github.com/rayon-rs/rayon) framework to parallelize some computations. By default, **Rayon** spawns the same number of threads as the number of CPUs (logicals cores) available. Instead of making things faster, the penalty of switching between so many threads may severely hurt the performance, so it's strongly advised to keep the number of threads small by manually setting `RAYON_NUM_THREADS` environment variable.

```
# empirically found to be the sweet spot for the number of threads
export RAYON_NUM_THREADS=2
cargo run --release --example image_demo model/seeta_fd_frontal_v1.0.bin <path-to-image>
```

## TODO

* Parallelize remaining CPU intensive loops
* Tests (it would make sense to start with an integration test for `Detector::detect`, based on the results retrieved from the original library)

## License

Original SeetaFace Detection is released under the [BSD 2-Clause license](https://github.com/seetaface/SeetaFaceEngine/blob/master/LICENSE). This project is a derivative work and uses the same license as the original.