extern crate rustface;
extern crate opencv;

use std::env::Args;

use opencv::core::Mat;
use opencv::highgui::{imread, IMREAD_UNCHANGED};
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

    let mut image: Mat = match imread(&options.image_path(), IMREAD_UNCHANGED) {
        Ok(image) => image,
        Err(message) => {
            println!("Failed to read image: {}", message);
            std::process::exit(1);
        }
    };

    if image.channels().unwrap() != 1 {
        cvt_color(&image, &image, COLOR_BGR2GRAY, 0).expect("Failed to convert image to gray scale");
    }

    let image_size = image.size().unwrap();
    let mut image = ImageData::new(image.ptr0(0).unwrap(), image_size.width as u32, image_size.height as u32);
    detector.detect(&mut image);
}

struct Options {
    image_path: String,
    model_path: String,
}

impl Options {
    fn parse(args: Args) -> Result<Self, String> {
        let args: Vec<String> = args.into_iter().collect();
        if args.len() != 5 {
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