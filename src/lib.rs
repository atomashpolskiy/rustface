extern crate byteorder;

mod common;
mod math;
mod feat;
mod classifier;
pub mod model;

use std::{cmp, ptr};
use common::{FaceInfo, ImageData, ImagePyramid, Rectangle};
use model::Model;

trait Detector {
    fn detect(&mut self, image: &mut ImagePyramid) -> Vec<FaceInfo>;
}

struct FuStDetector {
    model: Model,
    wnd_data_buf: Vec<u8>,
    wnd_data: Vec<u8>,
    wnd_size: u32,
    slide_wnd_step_x: u32,
    slide_wnd_step_y: u32,
}

impl FuStDetector {
    fn new(model: Model) -> Self {
        let wnd_size = 40;
        let slide_wnd_step_x = 4;
        let slide_wnd_step_y = 4;
        let num_hierarchy = 0;

        FuStDetector {
            model,
            wnd_data_buf: Vec::with_capacity((wnd_size * wnd_size) as usize),
            wnd_data: Vec::with_capacity((wnd_size * wnd_size) as usize),
            wnd_size,
            slide_wnd_step_x,
            slide_wnd_step_y,
        }
    }

    fn set_window_size(&mut self, wnd_size: u32) {
        if wnd_size >= 20 {
            self.wnd_size = wnd_size;
        }
    }

    fn set_slide_window_step(&mut self, step_x: u32, step_y: u32) {
        if step_x > 0 {
            self.slide_wnd_step_x = step_x;
        }
        if step_y > 0 {
            self.slide_wnd_step_y = step_y;
        }
    }

    fn get_window_data(&mut self, img: &ImageData, wnd: &mut Rectangle) {
        let roi = wnd;

        let roi_width = roi.width() as i32;
        let roi_height = roi.height() as i32;
        let img_width = img.width() as i32;
        let img_height = img.height() as i32;

        let pad_right = cmp::max(roi.x() + roi_width - img_width, 0);
        let pad_left = if roi.x() >= 0 { 0 } else {
            let t = roi.x();
            roi.set_x(0);
            -t
        };
        let pad_bottom = cmp::max(roi.y() + roi_height - img_height, 0);
        let pad_top = if roi.y() >= 0 { 0 } else {
            let t = roi.y();
            roi.set_y(0);
            -t
        };

        self.wnd_data_buf.resize((roi_width * roi_height) as usize, 0);
        let mut src;
        unsafe {
            src = img.data().offset((roi.y() * img_width + roi.x()) as isize);
        }
        let mut dest = self.wnd_data_buf.as_mut_ptr();
        let len = roi_width as usize;
        let len2 = (roi_width - pad_left - pad_right) as usize;

        if pad_top > 0 {
            unsafe {
                ptr::write_bytes(dest, 0, len * pad_top as usize);
                dest = dest.offset((roi_width * pad_top) as isize);
            }
        }

        match (pad_left, pad_right) {
            (0, 0) => {
                for y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::copy_nonoverlapping(src, dest, len);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                    }
                }
            },
            (0, _) => {
                for y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::copy_nonoverlapping(src, dest, len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                        ptr::write_bytes(dest.offset(-pad_right as isize), 0, pad_right as usize);
                    }
                }
            },
            (_, 0) => {
                for y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::write_bytes(dest, 0, pad_left as usize);
                        ptr::copy_nonoverlapping(src, dest.offset(pad_left as isize), len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                    }
                }
            },
            (_, _) => {
                for y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::write_bytes(dest, 0, pad_left as usize);
                        ptr::copy_nonoverlapping(src, dest.offset(pad_left as isize), len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                        ptr::write_bytes(dest.offset(-pad_right as isize), 0, pad_right as usize);
                    }
                }
            },
        }

        if pad_bottom > 0 {
            unsafe {
                ptr::write_bytes(dest, 0, len * pad_bottom as usize);
            }
        }

        let src_img = ImageData::new(self.wnd_data_buf.as_ptr(), roi.width(), roi.height());
        common::resize_image(&src_img, self.wnd_data.as_mut_ptr(), self.wnd_size, self.wnd_size);
    }
}

impl Detector for FuStDetector {
    fn detect(&mut self, image: &mut ImagePyramid) -> Vec<FaceInfo> {
        let mut scale_factor = 0.0;
        let image_scaled_optional = image.get_next_scale_image(&mut scale_factor);

        let mut wnd_info = FaceInfo::new();
        let hierarchy_sizes = self.model.get_hierarchy_sizes();
        let proposals: Vec<Vec<FaceInfo>> = Vec::with_capacity(hierarchy_sizes[0] as usize);

        let classifiers = self.model.get_classifiers();
        let ref mut first_classifier = classifiers[0];

        while let Some(ref image_scaled) = image_scaled_optional {
            first_classifier.compute(image_scaled);

            let width = (self.wnd_size as f32 / scale_factor + 0.5) as u32;
            wnd_info.bbox_mut().set_width(width);
            wnd_info.bbox_mut().set_height(width);

            let mut x = 0;
            let mut y = 0;
            let max_x = image_scaled.width() - self.wnd_size;
            let max_y = image_scaled.height() - self.wnd_size;

            while y <= max_y {
                while x <= max_x {
                    first_classifier.set_roi(Rectangle::new(x as i32, y as i32, self.wnd_size, self.wnd_size));

                    wnd_info.bbox_mut().set_x((x as f32 / scale_factor + 0.5) as i32);
                    wnd_info.bbox_mut().set_y((y as f32 / scale_factor + 0.5) as i32);

                    for i in 0..hierarchy_sizes[0] as usize {
                        let score = (&*classifiers[i]).classify();
                    }
                    x += self.slide_wnd_step_x;
                }
                y += self.slide_wnd_step_y;
            }
        }

/*
while (img_scaled != nullptr) {
    feat_map_1->Compute(img_scaled->data, img_scaled->width,
      img_scaled->height);

    wnd_info.bbox.width = static_cast<int32_t>(wnd_size_ / scale_factor + 0.5);
    wnd_info.bbox.height = wnd_info.bbox.width;

    int32_t max_x = img_scaled->width - wnd_size_;
    int32_t max_y = img_scaled->height - wnd_size_;
    for (int32_t y = 0; y <= max_y; y += slide_wnd_step_y_) {
      wnd.y = y;
      for (int32_t x = 0; x <= max_x; x += slide_wnd_step_x_) {
        wnd.x = x;
        feat_map_1->SetROI(wnd);

        wnd_info.bbox.x = static_cast<int32_t>(x / scale_factor + 0.5);
        wnd_info.bbox.y = static_cast<int32_t>(y / scale_factor + 0.5);

        for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
          if (model_[i]->Classify(&score)) {
            wnd_info.score = static_cast<double>(score);
            proposals[i].push_back(wnd_info);
          }
        }
      }
    }

    img_scaled = img_pyramid->GetNextScaleImage(&scale_factor);
  }

*/








        vec![]
    }
}
