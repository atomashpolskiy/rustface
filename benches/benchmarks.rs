#[macro_use]
extern crate criterion;

use criterion::Criterion;

use criterion::{Bencher, Benchmark, Fun};
use image::DynamicImage;
use rustface::math::{abs, square, vector_add, vector_inner_product, vector_sub};
use rustface::{Detector, ImageData};
use std::time::Duration;

fn get_test_image() -> DynamicImage {
    let test_img_path = "./assets/test/scientists.jpg";
    let image: DynamicImage = match image::open(test_img_path) {
        Ok(image) => image,
        Err(message) => {
            println!("Failed to read image: {}", message);
            std::process::exit(1)
        }
    };

    return image;
}

fn get_default_detector() -> Box<dyn Detector> {
    let model_path = "./model/seeta_fd_frontal_v1.0.bin";
    let mut detector = match rustface::create_detector(model_path) {
        Ok(detector) => detector,
        Err(error) => {
            println!("Failed to create detector: {}", error.to_string());
            std::process::exit(1)
        }
    };

    // configure detector
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    return detector;
}

fn detect_single_image(c: &mut Criterion) {
    let mut detector = get_default_detector();
    let img = get_test_image().to_luma8();

    // convert to rustface internal image datastructure
    let (width, height) = img.dimensions();

    let target_runtime = Duration::new(100, 0);

    c.bench(
        "detect_single_image",
        Benchmark::new("detect", move |b| {
            let test_image = ImageData::new(&img, width, height);
            b.iter(|| detector.detect(&test_image))
        })
        // Limit the measurement time and the sample size
        // to make sure the benchmark finishes in a feasible amount of time.
        .measurement_time(target_runtime)
        .sample_size(20),
    );
}

fn bench_square(c: &mut Criterion) {
    c.bench_function("math_square 500", |b| {
        let src: Vec<_> = (0..500).map(|i| i).collect();
        let mut dest = vec![0; 500];
        b.iter(|| {
            square(&src, &mut dest);
        })
    });
}

fn bench_abs(c: &mut Criterion) {
    c.bench_function("math_abs 500", move |b| {
        let vec: Vec<_> = (0..500).map(|i| if i % 3 == 0 { i } else { -i }).collect();
        let mut dest = vec![0; 500];
        b.iter(|| {
            unsafe { abs(vec.as_ptr(), dest.as_mut_ptr(), vec.len()) };
        })
    });
}

fn bench_vector_add_dest(c: &mut Criterion) {
    c.bench_function("math_vector_add dest", move |b| {
        let src: Vec<_> = (0..500).map(|i| i).collect();
        let mut dest = vec![0; 500];
        b.iter(|| {
            unsafe { vector_add(src.as_ptr(), src.as_ptr(), dest.as_mut_ptr(), src.len()) };
        })
    });
}

fn bench_vector_add_inplace(c: &mut Criterion) {
    c.bench_function("math_vector_add in-place", move |b| {
        let mut vec: Vec<_> = (0..500).map(|i| i).collect();
        b.iter(|| {
            unsafe { vector_add(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len()) };
        })
    });
}

fn bench_vector_sub_inplace(c: &mut Criterion) {
    c.bench_function("math_vector_sub in-place", move |b| {
        let mut vec: Vec<_> = (0..500).map(|i| i).collect();
        b.iter(|| {
            unsafe { vector_sub(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len()) };
        })
    });
}

fn bench_vector_inner_product(c: &mut Criterion) {
    c.bench_function("math_vector_inner_product 500", move |b| {
        let vec: Vec<_> = (0..500).map(|i| i as f32).collect();
        b.iter(|| {
            vector_inner_product(&vec, &vec);
        })
    });
}

fn bench_square_compare(c: &mut Criterion) {
    let naive = Fun::new("naive", |b: &mut Bencher<'_>, input: &Vec<i32>| {
        let mut target: Vec<i32> = vec![0; input.len()];
        b.iter(|| {
            for (i, e) in input.iter().enumerate() {
                target[i] = e.pow(2);
            }
        })
    });

    let naive_iterator = Fun::new("naive_iterator", |b: &mut Bencher<'_>, input: &Vec<i32>| {
        b.iter(|| {
            let _target: Vec<i32> = input.iter().map(|a| a.pow(2)).collect();
        })
    });

    let notsafe = Fun::new("unsafe", |b, input: &Vec<i32>| {
        let mut target: Vec<u32> = vec![0; input.len()];
        b.iter(|| {
            square(&input, &mut target[..input.len()]);
        })
    });

    let testvec_size: usize = 1000;
    let mut testvec = Vec::<i32>::with_capacity(testvec_size);
    for i in 0..testvec_size {
        testvec.push(i as i32);
    }

    let functions = vec![naive, naive_iterator, notsafe];

    c.bench_functions("square_comparison", functions, testvec);
}

fn bench_abs_compare(c: &mut Criterion) {
    let naive = Fun::new("naive", |b: &mut Bencher<'_>, input: &Vec<i32>| {
        let mut target: Vec<i32> = vec![0; input.len()];
        b.iter(|| {
            for (i, e) in input.iter().enumerate() {
                target[i] = e.abs();
            }
        })
    });

    let naive_iterator = Fun::new("naive_iterator", |b: &mut Bencher<'_>, input: &Vec<i32>| {
        b.iter(|| {
            let _target: Vec<i32> = input.iter().map(|a| a.abs()).collect();
        })
    });

    let notsafe = Fun::new("unsafe", |b, input: &Vec<i32>| {
        let mut target: Vec<i32> = vec![0; input.len()];
        b.iter(|| {
            unsafe { abs(input.as_ptr(), target.as_mut_ptr(), input.len()) };
        })
    });

    let testvec_size: usize = 1000;
    let mut testvec = Vec::<i32>::with_capacity(testvec_size);
    for i in 0..testvec_size {
        testvec.push(i as i32);
    }

    let functions = vec![naive, naive_iterator, notsafe];

    c.bench_functions("abs_comparison", functions, testvec);
}

criterion_group!(detection_perf, detect_single_image);
criterion_group!(
    math,
    bench_square,
    bench_abs,
    bench_vector_add_dest,
    bench_vector_add_inplace,
    bench_vector_sub_inplace,
    bench_vector_inner_product
);
criterion_group!(math_compare, bench_square_compare, bench_abs_compare);
criterion_main!(detection_perf, math, math_compare);
