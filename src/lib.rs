extern crate byteorder;

mod common;
mod math;
mod feat;
mod classifier;
pub mod model;

use std::{cmp, ptr};
use std::cell::RefCell;
use std::rc::Rc;
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
        let mut image_scaled_optional = image.get_next_scale_image(&mut scale_factor);

        let wnd_info = Rc::new(RefCell::new(FaceInfo::new()));
        let first_hierarchy_size = self.model.get_hierarchy_size(0) as usize;
        let mut proposals: Vec<Vec<Rc<RefCell<FaceInfo>>>> = Vec::with_capacity(first_hierarchy_size);

        loop {
            match image_scaled_optional {
                Some(ref image_scaled) => {
                    self.model.get_classifiers()[0].compute(image_scaled);

                    let width = (self.wnd_size as f32 / scale_factor + 0.5) as u32;
                    wnd_info.borrow_mut().bbox_mut().set_width(width);
                    wnd_info.borrow_mut().bbox_mut().set_height(width);

                    let mut x = 0;
                    let mut y = 0;
                    let max_x = image_scaled.width() - self.wnd_size;
                    let max_y = image_scaled.height() - self.wnd_size;

                    while y <= max_y {
                        while x <= max_x {
                            self.model.get_classifiers()[0].set_roi(Rectangle::new(x as i32, y as i32, self.wnd_size, self.wnd_size));

                            wnd_info.borrow_mut().bbox_mut().set_x((x as f32 / scale_factor + 0.5) as i32);
                            wnd_info.borrow_mut().bbox_mut().set_y((y as f32 / scale_factor + 0.5) as i32);

                            for i in 0..first_hierarchy_size {
                                let score = (&mut *self.model.get_classifiers()[i]).classify(None);
                                if score.is_positive() {
                                    wnd_info.borrow_mut().set_score(score.score() as f64);
                                    proposals[i].push(Rc::clone(&wnd_info));
                                }
                            }
                            x += self.slide_wnd_step_x;
                        }
                        y += self.slide_wnd_step_y;
                    }
                },
                None => break,
            }
            image_scaled_optional = image.get_next_scale_image(&mut scale_factor);
        }



        /*
        std::vector<std::vector<seeta::FaceInfo> > proposals_nms(hierarchy_size_[0]);
          for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
            seeta::fd::NonMaximumSuppression(&(proposals[i]),
              &(proposals_nms[i]), 0.8f);
            proposals[i].clear();
          }

        */







        vec![]
    }
}
