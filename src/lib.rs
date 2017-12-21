extern crate byteorder;

mod common;
mod math;
mod feat;
mod classifier;
pub mod model;

use std::{cmp, ptr};
use common::{FaceInfo, ImageData, ImagePyramid, Rectangle};

trait Detector {
    fn detect(&mut self, image: &ImagePyramid) -> Vec<FaceInfo>;
}

struct FuStDetector {
    wnd_data_buf: Vec<u8>,
    wnd_data: Vec<u8>,
    wnd_size: u32,
    slide_wnd_step_x: i32,
    slide_wnd_step_y: i32,
    num_hierarchy: u32,
}

impl FuStDetector {
    fn new() -> Self {
        let wnd_size = 40;
        let slide_wnd_step_x = 4;
        let slide_wnd_step_y = 4;
        let num_hierarchy = 0;

        FuStDetector {
            wnd_data_buf: Vec::with_capacity((wnd_size * wnd_size) as usize),
            wnd_data: Vec::with_capacity((wnd_size * wnd_size) as usize),
            wnd_size,
            slide_wnd_step_x,
            slide_wnd_step_y,
            num_hierarchy,
        }
    }

    fn set_window_size(&mut self, wnd_size: u32) {
        if size >= 20 {
            self.wnd_size = wnd_size;
        }
    }

    fn set_slide_window_step(&mut self, step_x: i32, step_y: i32) {
        if step_x > 0 {
            self.slide_wnd_step_x = step_x;
        }
        if step_y > 0 {
            self.slide_wnd_step_y = step_y;
        }
    }

    fn get_window_data(&mut self, img: &ImageData, wnd: &mut Rectangle) {
        let mut roi = wnd;

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
