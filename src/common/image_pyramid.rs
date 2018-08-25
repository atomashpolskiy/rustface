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

use std::cmp;
use std::ptr;

#[derive(Debug)]
pub struct ImageData {
    data: *const u8,
    width: u32,
    height: u32,
    num_channels: u32,
}

impl ImageData {
    pub fn new(data: *const u8, width: u32, height: u32) -> Self {
        ImageData {
            data,
            width,
            height,
            num_channels: 1,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn num_channels(&self) -> u32 {
        self.num_channels
    }

    fn copy_to(&self, dest: *mut u8) {
        unsafe {
            ptr::copy_nonoverlapping(self.data, dest, (self.width * self.height) as usize);
        }
    }

    pub fn data(&self) -> *const u8 {
        self.data
    }
}

pub struct ImagePyramid {
    max_scale: f32,
    min_scale: f32,
    scale_factor: f32,
    scale_step: f32,
    width1x: u32,
    height1x: u32,
    width_scaled: u32,
    height_scaled: u32,
    img_buf: Vec<u8>,
    img_buf_width: u32,
    img_buf_height: u32,
    img_buf_scaled: Vec<u8>,
    img_buf_scaled_width: u32,
    img_buf_scaled_height: u32,
}

impl ImagePyramid {
    pub fn new() -> Self {
        let img_buf_width: u32 = 2;
        let img_buf_height: u32 = 2;
        let img_buf_scaled_width: u32 = 2;
        let img_buf_scaled_height: u32 = 2;

        ImagePyramid {
            max_scale: 1.0,
            min_scale: 1.0,
            scale_factor: 1.0,
            scale_step: 0.8,
            width1x: 0,
            height1x: 0,
            width_scaled: 0,
            height_scaled: 0,
            img_buf: Vec::with_capacity((img_buf_width * img_buf_height) as usize),
            img_buf_width,
            img_buf_height,
            img_buf_scaled: Vec::with_capacity(
                (img_buf_scaled_width * img_buf_scaled_height) as usize,
            ),
            img_buf_scaled_width,
            img_buf_scaled_height,
        }
    }

    #[allow(dead_code)]
    pub fn set_max_scale(&mut self, max_scale: f32) {
        self.max_scale = max_scale;
        self.scale_factor = max_scale;
        self.update_buf_scaled();
    }

    pub fn set_min_scale(&mut self, min_scale: f32) {
        self.min_scale = min_scale;
    }

    pub fn set_scale_step(&mut self, scale_step: f32) {
        if scale_step > 0.0 && scale_step <= 1.0 {
            self.scale_step = scale_step;
        }
    }

    pub fn get_image_1x(&self) -> ImageData {
        ImageData::new(self.img_buf.as_ptr(), self.width1x, self.height1x)
    }

    pub fn set_image_1x(&mut self, img_data: *const u8, width: u32, height: u32) {
        if width > self.img_buf_width || height > self.img_buf_height {
            self.img_buf_width = width;
            self.img_buf_height = height;
            self.img_buf = Vec::with_capacity((width * height) as usize);
        }

        self.width1x = width;
        self.height1x = height;

        unsafe {
            ptr::copy_nonoverlapping(
                img_data,
                self.img_buf.as_mut_ptr(),
                (width * height) as usize,
            );
        }

        self.scale_factor = self.max_scale;
        self.update_buf_scaled();
    }

    fn update_buf_scaled(&mut self) {
        if self.width1x == 0 || self.height1x == 0 {
            return;
        }

        let max_width = (self.width1x as f32 * self.max_scale + 0.5) as u32;
        let max_height = (self.height1x as f32 * self.max_scale + 0.5) as u32;

        if max_width > self.img_buf_scaled_width || max_height > self.img_buf_scaled_height {
            self.img_buf_scaled_width = max_width;
            self.img_buf_scaled_height = max_height;
            self.img_buf_scaled = Vec::with_capacity((max_width * max_height) as usize);
        }
    }

    pub fn get_next_scale_image(&mut self, scale_factor: &mut f32) -> Option<ImageData> {
        if self.scale_factor < self.min_scale {
            return None;
        }

        *scale_factor = self.scale_factor;
        self.width_scaled = (self.width1x as f32 * self.scale_factor) as u32;
        self.height_scaled = (self.height1x as f32 * self.scale_factor) as u32;

        let src = ImageData::new(self.img_buf.as_ptr(), self.width1x, self.height1x);
        resize_image(
            &src,
            self.img_buf_scaled.as_mut_ptr(),
            self.width_scaled,
            self.height_scaled,
        );
        let img_scaled = Some(ImageData::new(
            self.img_buf_scaled.as_ptr(),
            self.width_scaled,
            self.height_scaled,
        ));
        self.scale_factor *= self.scale_step;

        img_scaled
    }
}

pub fn resize_image(src: &ImageData, dest: *mut u8, width: u32, height: u32) {
    if src.width() == width && src.height() == height {
        src.copy_to(dest);
        return;
    }

    let src_data = src.data();

    let lf_x_scl = f64::from(src.width()) / f64::from(width);
    let lf_y_scl = f64::from(src.height()) / f64::from(height);

    unsafe {
        for y in 0..height {
            for x in 0..width {
                let lf_x_s = lf_x_scl * f64::from(x);
                let lf_y_s = lf_y_scl * f64::from(y);

                let n_x_s = cmp::min(lf_x_s as u32, src.width() - 2);
                let n_y_s = cmp::min(lf_y_s as u32, src.height() - 2);

                let lf_weight_x = lf_x_s - f64::from(n_x_s);
                let lf_weight_y = lf_y_s - f64::from(n_y_s);

                let d1 = f64::from(*src_data.offset((n_y_s * src.width() + n_x_s) as isize));
                let d2 = f64::from(*src_data.offset((n_y_s * src.width() + n_x_s + 1) as isize));
                let d3 = f64::from(*src_data.offset(((n_y_s + 1) * src.width() + n_x_s) as isize));
                let d4 =
                    f64::from(*src_data.offset(((n_y_s + 1) * src.width() + n_x_s + 1) as isize));

                let dest_val = (1.0 - lf_weight_y) * ((1.0 - lf_weight_x) * d1 + lf_weight_x * d2)
                    + lf_weight_y * ((1.0 - lf_weight_x) * d3 + lf_weight_x * d4);

                *dest.offset((y * width + x) as isize) = dest_val as u8;
            }
        }
    }
}
