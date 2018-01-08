// This file is part of the open-source port of SeetaFace engine, which originally includes three modules:
//      SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
//
// This file is part of the SeetaFace Detection module, containing codes implementing the face detection method described in the following paper:
//
//      Funnel-structured cascade for multi-view face detection with alignment awareness,
//      Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
//      In Neurocomputing (under review)
//
// Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
// Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
//
// As an open-source face recognition engine: you can redistribute SeetaFace source codes
// and/or modify it under the terms of the BSD 2-Clause License.
//
// You should have received a copy of the BSD 2-Clause License along with the software.
// If not, see < https://opensource.org/licenses/BSD-2-Clause>.

extern crate rustface;
extern crate opencv;
extern crate cpuprofiler;

use std::env::Args;
use std::time::{Duration, Instant};

use opencv::core::{Mat, Rect, rectangle, Scalar};
use opencv::highgui::{destroy_all_windows, imread, IMREAD_UNCHANGED, imshow, named_window, wait_key, WINDOW_AUTOSIZE};
use opencv::imgproc::{cvt_color, COLOR_BGR2GRAY};

use cpuprofiler::PROFILER;

use rustface::{Detector, FaceInfo, ImageData};

fn main() {
    let options = match Options::parse(std::env::args()) {
        Ok(options) => options,
        Err(message) => {
            println!("Failed to parse program arguments: {}", message);
            std::process::exit(1);
        }
    };

    let mut detector = match rustface::create_detector(options.model_path()) {
        Ok(detector) => detector,
        Err(error) => {
            println!("Failed to create detector: {}", error.to_string());
            std::process::exit(1);
        }
    };

    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    let mut mat: Mat = match imread(&options.image_path(), IMREAD_UNCHANGED) {
        Ok(image) => image,
        Err(message) => {
            println!("Failed to read image: {}", message);
            std::process::exit(1);
        }
    };

    let faces;
    if mat.channels().unwrap() != 1 {
        let mut mat_gray = Mat::new().unwrap();
        cvt_color(&mat, &mat_gray, COLOR_BGR2GRAY, 0).expect("Failed to convert image to gray scale");
        faces = detect_faces(&mut detector, &mut mat_gray);
    } else {
        faces = detect_faces(&mut detector, &mut mat);
    }

    for face in faces.into_iter() {
        let rect = Rect {
            x: face.bbox().x(),
            y: face.bbox().y(),
            width: face.bbox().width() as i32,
            height: face.bbox().height() as i32,
        };
        rectangle(&mat, rect, Scalar {data: [0.0, 0.0, 255.0, 0.0]}, 4, 8, 0).unwrap();
    }

    named_window("Test", WINDOW_AUTOSIZE).unwrap();
    imshow("Test", &mat).unwrap();
    wait_key(0).unwrap();
    destroy_all_windows().unwrap();
}

fn detect_faces(detector: &mut Box<Detector>, mat: &mut Mat) -> Vec<FaceInfo> {
    let image_size = mat.size().unwrap();
    let mut image = ImageData::new(mat.ptr0(0).unwrap(), image_size.width as u32, image_size.height as u32);
    let now = Instant::now();
    // uncomment to profile
    // PROFILER.lock().unwrap().start("./opencv_demo.profile").unwrap();
    let faces = detector.detect(&mut image);
    // PROFILER.lock().unwrap().stop().unwrap();
    println!("Found {} faces in {} ms", faces.len(), get_millis(now.elapsed()));
    faces
}

fn get_millis(duration: Duration) -> u64 {
    duration.as_secs() * 1000u64 + (duration.subsec_nanos() / 1_000_000) as u64
}

struct Options {
    image_path: String,
    model_path: String,
}

impl Options {
    fn parse(args: Args) -> Result<Self, String> {
        let args: Vec<String> = args.into_iter().collect();
        if args.len() != 3 {
            return Err(format!("Usage: {} <model-path> <image-path>", args[0]))
        }

        let model_path = args[1].clone();
        let image_path = args[2].clone();

        Ok(Options { image_path, model_path })
    }

    fn image_path(&self) -> &str {
        &self.image_path[..]
    }

    fn model_path(&self) -> &str {
        &self.model_path[..]
    }
}