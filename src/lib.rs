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

mod classifier;
mod common;
mod detector;
mod feat;
pub mod math;
pub mod model;

pub use crate::common::FaceInfo;
pub use crate::common::ImageData;
pub use crate::common::Rectangle;
pub use crate::model::{load_model, read_model, Model};

use crate::detector::FuStDetector;
use std::io;

/// Create a face detector, based on a file with model description.
pub fn create_detector(path_to_model: &str) -> Result<Box<dyn Detector>, io::Error> {
    let model = load_model(path_to_model)?;
    Ok(create_detector_with_model(model))
}

/// Create a face detector, based on the provided model.
pub fn create_detector_with_model(model: Model) -> Box<dyn Detector> {
    Box::new(FuStDetector::new(model))
}

pub trait Detector {
    /// Detect faces on input image.
    ///
    /// (1) The input image should be gray-scale, i.e. `num_channels` set to 1.
    /// (2) Currently this function does not give the Euler angles, which are
    ///     left with invalid values.
    ///
    /// # Panics
    ///
    /// Panics if `image` is not a legal image, e.g. it
    /// - is not gray-scale (`num_channels` is not equal to 1)
    /// - has `width` or `height` equal to 0
    fn detect(&mut self, image: &ImageData) -> Vec<FaceInfo>;

    /// Set the size of the sliding window.
    ///
    /// The minimum size is constrained as no smaller than 20.
    ///
    /// # Panics
    ///
    /// Panics if `wnd_size` is less than 20.
    fn set_window_size(&mut self, wnd_size: u32);

    /// Set the sliding window step in horizontal and vertical directions.
    ///
    /// The steps should take positive values.
    /// Usually a step of 4 is a reasonable choice.
    ///
    /// # Panics
    ///
    /// Panics if `step_x` or `step_y` is less than or equal to 0.
    fn set_slide_window_step(&mut self, step_x: u32, step_y: u32);

    /// Set the minimum size of faces to detect.
    ///
    /// The minimum size is constrained as no smaller than 20.
    ///
    /// # Panics
    ///
    /// Panics if `min_face_size` is less than 20.
    fn set_min_face_size(&mut self, min_face_size: u32);

    /// Set the maximum size of faces to detect.
    ///
    /// The maximum face size actually used is computed as the minimum among:
    /// user specified size, image width, image height.
    fn set_max_face_size(&mut self, max_face_size: u32);

    /// Set the factor between adjacent scales of image pyramid.
    ///
    /// The value of the factor lies in (0.1, 0.99). For example, when it is set as 0.5,
    /// an input image of size w x h will be resized to 0.5w x 0.5h, 0.25w x 0.25h,  0.125w x 0.125h, etc.
    ///
    /// # Panics
    ///
    /// Panics if `scale_factor` is less than 0.01 or greater than 0.99
    fn set_pyramid_scale_factor(&mut self, scale_factor: f32);

    /// Set the score threshold of detected faces.
    ///
    /// Detections with scores smaller than the threshold will not be returned.
    /// Typical threshold values include 0.95, 2.8, 4.5. One can adjust the
    /// threshold based on his or her own test set.
    ///
    /// Smaller values result in more detections (possibly increasing the number of false positives),
    /// larger values result in fewer detections (possibly increasing the number of false negatives).
    ///
    /// # Panics
    ///
    /// Panics if `thresh` is less than or equal to 0.
    fn set_score_thresh(&mut self, thresh: f64);
}
