<h1 align="center">
    <a href="http://atomashpolskiy.github.io/rustface/">Rustface</a>
</h1>

<p align="center"><strong>
<sup>
<br/><a href="https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceDetection">SeetaFace Detection</a> in Rust programming language
</sup>
</strong></p>

<p align="center">
    <img src="https://atomashpolskiy.github.io/static/img/scientists.png" alt="Bt Example">
</p>

## API

```rust
extern crate rustface;

use rustface::{Detector, ImageData};

fn main() {
    let mut detector = rustface::create_detector("/path/to/model").unwrap();
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);
    
    let mut image = ImageData::new(bytes, width, height);
    for face in detector.detect(&mut image).into_iter() {
    
        println!("x: {}, y: {}, w")
    };
}
```

## How to build

The project is a library crate, but also contains a runnable module for demonstration purposes. In order to build it, you'll need an OpenCV 2.4 installation for [generation of Rust bindings](https://github.com/kali/opencv-rust).

Otherwise, the project relies on the stable Rust toolchain, so just use the standard Cargo `build` command:

```
cargo build --release
```

## Run demo

Code for the demo is located in `main.rs` file. It performs face detection for the given image and opens it in a separate window.

```
$ cargo run --release model/seeta_fd_frontal_v1.0.bin <path-to-image>
```

## TODO

* Use SSE for certain transformations in `math` module and `SurfMlpFeatureMap`
* Use OpenMP for parallelization of some CPU intensive loops
* Tests (it would make sense to start with an integration test for `Detector::detect`, based on the results retrieved from the original library)

## License

Original SeetaFace Detection is released under the [BSD 2-Clause license](https://github.com/seetaface/SeetaFaceEngine/blob/master/LICENSE). This project is a derivative work and uses the same license as the original.