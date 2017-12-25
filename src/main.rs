extern crate rustface;
extern crate opencv;

use std::env::Args;

use opencv::core::{Mat, Rect, rectangle, Scalar};
use opencv::highgui::{destroy_all_windows, imread, IMREAD_UNCHANGED, imshow, named_window, wait_key, WINDOW_AUTOSIZE};
use opencv::imgproc::{cvt_color, COLOR_BGR2GRAY};

use rustface::ImageData;

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

    if mat.channels().unwrap() != 1 {
        cvt_color(&mat, &mat, COLOR_BGR2GRAY, 0).expect("Failed to convert image to gray scale");
    }

    let image_size = mat.size().unwrap();
    let mut image = ImageData::new(mat.ptr0(0).unwrap(), image_size.width as u32, image_size.height as u32);

    let faces = detector.detect(&mut image);
    println!("Found {} faces", faces.len());
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