
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

We want the current master branch to be covered by stability guarantees, so we have moved the code that relies on unstable explicit SIMD intrinsics to the `nightly` branch. 

Benchmarks for the `master` branch are coming soon.

### Using nightly Rust

The `nightly` branch contains a slightly (~20%) faster version of rustface. This speedup is made possible by using explicit SIMD intrinsics.  If you want to use this branch, you need an older nightly toolchain.

```
rustup toolchain install nightly-2018-01-15
rustup default nightly-2018-01-15
```

Regarding the performance of the `nightly` branch: crude manual benchmarking showed that this nightly Rust version of SeetaFace is _slightly faster_ than the original C++ version. In this particular test the Rust version has been **4% faster on average** than its C++ counterpart. When using multiple threads and enabling LTO (link-time optimization), Rust performance is a tad better (I observe a **8% boost**):

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

* Use stable SIMD intrinsics when available
* Parallelize remaining CPU intensive loops
* Tests (it would make sense to start with an integration test for `Detector::detect`, based on the results retrieved from the original library)

## Authors

- Andrei Tomashpolskiy \<nordmann89@gmail.com\>
  
  _Original developer and maintainer_
  
- Ashley \<[github.com/expenses](https://github.com/expenses)\>

  _Contributed the switch from OpenCV to [Image](https://crates.io/crates/image)_

This library is based on the following works:

- Face detection method described in the paper: _"Funnel-structured cascade for multi-view face detection with alignment awareness, Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen. In Neurocomputing (under review)"_

- [original C++ implementation](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceDetection)

## License

Original SeetaFace Detection is released under the [BSD 2-Clause license](https://github.com/seetaface/SeetaFaceEngine/blob/master/LICENSE). This project is a derivative work and uses the same license as the original.