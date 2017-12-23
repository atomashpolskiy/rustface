extern crate rustface;
//extern crate opencv;

use std::env::Args;
use std::fs::File;
use std::io;
use std::io::{Error, Read};

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

    let mut buf = Vec::new();
    if let Err(error) = options.image().read_to_end(&mut buf) {
        println!("Failed to read image: {}", error.to_string());
        std::process::exit(1);
    }
    let mut image = ImageData::new(buf.as_ptr(), options.image_width(), options.image_height());
    let faceinfo = detector.detect(&mut image);
}

struct Options {
    image: File,
    model_path: String,
    image_width: u32,
    image_height: u32,
}

impl Options {
    fn parse(args: Args) -> Result<Self, String> {
        let args: Vec<String> = args.into_iter().collect();
        if args.len() != 5 {
            return Err(format!("Usage: {} <model-path> <image-path> <width> <height>", args[0]))
        }

        let model_path = args[1].clone();
        let image_path = Options::open_file(&args[2])?;
        let image_width: u32 = Options::parse_int(&args[3])? as u32;
        let image_height: u32 = Options::parse_int(&args[4])? as u32;

        Ok(Options { image: image_path, model_path, image_width, image_height })
    }

    fn open_file(path: &String) -> Result<File, String> {
        File::open(path).map_err(|e| e.to_string())
    }

    fn parse_int(s: &String) -> Result<i32, String> {
        s.parse::<i32>().map_err(|e| e.to_string())
    }

    fn image(&self) -> &File {
        &self.image
    }

    fn image_width(&self) -> u32 {
        self.image_width
    }

    fn image_height(&self) -> u32 {
        self.image_height
    }

    fn model_path(&self) -> &str {
        &self.model_path[..]
    }
}