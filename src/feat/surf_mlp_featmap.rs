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

use crate::common::{Rectangle, Seq};
use crate::feat::FeatureMap;
use crate::math;
use crate::ImageData;
use std::ptr;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub struct SurfMlpFeatureMap {
    width: u32,
    height: u32,
    length: usize,
    feature_pool: FeaturePool,
    feature_vectors: Vec<Vec<i32>>,
    feature_vectors_normalized: Vec<Vec<f32>>,
    grad_x: Vec<i32>,
    grad_y: Vec<i32>,
    int_img: Vec<i32>,
    img_buf: Vec<i32>,
}

impl FeatureMap for SurfMlpFeatureMap {
    fn compute(&mut self, image: &ImageData) {
        let input = image.data();
        let width = image.width();
        let height = image.height();

        if width == 0 || height == 0 {
            panic!("Illegal arguments: width ({}), height ({})", width, height);
        }

        self.reshape(width, height);
        self.compute_gradient_images(input);
        self.compute_integral_images();
    }
}

impl SurfMlpFeatureMap {
    pub fn new() -> Self {
        let feature_pool = SurfMlpFeatureMap::create_feature_pool();
        let feature_pool_size = feature_pool.size();
        let mut feature_vectors = Vec::with_capacity(feature_pool_size);
        let mut feature_vectors_normalized = Vec::with_capacity(feature_pool_size);
        for feature_id in 0..feature_pool_size {
            let dim = feature_pool.get_feature_vector_dim(feature_id);
            feature_vectors.push(vec![0; dim]);
            feature_vectors_normalized.push(vec![0.0; dim]);
        }

        SurfMlpFeatureMap {
            width: 0,
            height: 0,
            length: 0,
            feature_pool,
            feature_vectors,
            feature_vectors_normalized,
            grad_x: Vec::new(),
            grad_y: Vec::new(),
            int_img: Vec::new(),
            img_buf: Vec::new(),
        }
    }

    fn create_feature_pool() -> FeaturePool {
        let mut feature_pool = FeaturePool::new();
        feature_pool.add_patch_format(1, 1, 2, 2);
        feature_pool.add_patch_format(1, 2, 2, 2);
        feature_pool.add_patch_format(2, 1, 2, 2);
        feature_pool.add_patch_format(2, 3, 2, 2);
        feature_pool.add_patch_format(3, 2, 2, 2);
        feature_pool.create();
        feature_pool
    }

    fn reshape(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.length = width as usize * height as usize;

        self.grad_x.resize(self.length, 0);
        self.grad_y.resize(self.length, 0);
        self.int_img
            .resize(self.length * FeaturePool::K_NUM_INT_CHANNEL as usize, 0);
        self.img_buf.resize(self.length, 0);
    }

    fn compute_gradient_images(&mut self, input: &[u8]) {
        assert_eq!(input.len(), self.length);

        math::copy_u8_to_i32(input, &mut self.img_buf);
        self.compute_grad_x();
        self.compute_grad_y();
    }

    fn compute_grad_x(&mut self) {
        let input = self.img_buf.as_ptr();
        let dx = self.grad_x.as_mut_ptr();
        let len = (self.width - 2) as usize;

        unsafe {
            for r in 0..self.height {
                let offset = (r * self.width) as isize;
                let mut src = input.offset(offset);
                let mut dest = dx.offset(offset);
                *dest = ((*(src.offset(1))) - (*src)) << 1;
                math::vector_sub(src.offset(2), src, dest.offset(1), len);

                let offset = (self.width - 1) as isize;
                src = src.offset(offset);
                dest = dest.offset(offset);
                *dest = ((*src) - (*(src.offset(-1)))) << 1;
            }
        }
    }

    fn compute_grad_y(&mut self) {
        let input = self.img_buf.as_ptr();
        let mut dy = self.grad_y.as_mut_ptr();
        let len = self.width as usize;

        unsafe {
            math::vector_sub(input.offset(self.width as isize), input, dy, len);
            math::vector_add(dy, dy, dy, len);

            let step = self.width as usize;
            let grad_y_stop = self.grad_y.len() - step;

            #[cfg(feature = "rayon")]
            let it = self
                .img_buf
                .par_chunks(step)
                .zip(self.grad_y[step..grad_y_stop].par_chunks_mut(step));

            #[cfg(not(feature = "rayon"))]
            let it = self
                .img_buf
                .chunks(step)
                .zip(self.grad_y[step..grad_y_stop].chunks_mut(step));

            it.for_each(|(inputs, outputs)| {
                let src = inputs.as_ptr();
                let dest = outputs.as_mut_ptr();
                math::vector_sub(src.offset((step << 1) as isize), src, dest, len);
            });

            let offset = ((self.height - 1) * self.width) as isize;
            dy = dy.offset(offset);
            math::vector_sub(
                input.offset(offset),
                input.offset(offset - self.width as isize),
                dy,
                len,
            );
            math::vector_add(dy, dy, dy, len);
        }
    }

    fn compute_integral_images(&mut self) {
        let grad_x_ptr = self.grad_x.as_ptr();
        let grad_y_ptr = self.grad_y.as_ptr();
        let img_buf_ptr = self.img_buf.as_ptr();

        unsafe {
            self.fill_integral_channel(grad_x_ptr, 0);
            self.fill_integral_channel(grad_y_ptr, 4);
            math::abs(grad_x_ptr, img_buf_ptr as *mut i32, self.length);
            self.fill_integral_channel(img_buf_ptr, 1);
            math::abs(grad_y_ptr, img_buf_ptr as *mut i32, self.length);
            self.fill_integral_channel(img_buf_ptr, 5);
        }

        self.mask_integral_channel();
        self.integral();
    }

    unsafe fn fill_integral_channel(&mut self, mut src: *const i32, ch: u32) {
        let mut dest = self.int_img.as_mut_ptr().offset(ch as isize);
        for _ in 0..self.length {
            *dest = *src;
            *dest.offset(2) = *src;
            dest = dest.offset(FeaturePool::K_NUM_INT_CHANNEL as isize);
            src = src.offset(1);
        }
    }

    fn mask_integral_channel(&mut self) {
        self.mask_integral_channel_portable();
    }

    fn mask_integral_channel_portable(&mut self) {
        let mut grad_x_ptr = self.grad_x.as_ptr();
        let mut grad_y_ptr = self.grad_y.as_ptr();

        let mut dx: i32;
        let mut dy: i32;
        let mut dx_mask: i32;
        let mut dy_mask: i32;
        let mut cmp: u32;
        let xor_bits: Vec<u32> = vec![0xffff_ffff, 0xffff_ffff, 0, 0];

        let mut src = self.int_img.as_mut_ptr();
        unsafe {
            for _ in 0..self.length {
                dx = *grad_x_ptr;
                grad_x_ptr = grad_x_ptr.offset(1);
                dy = *grad_y_ptr;
                grad_y_ptr = grad_y_ptr.offset(1);

                cmp = if dy < 0 { 0xffff_ffff } else { 0x0 };
                for j in xor_bits.iter().take(4) {
                    dy_mask = (cmp ^ j) as i32;
                    *src &= dy_mask;
                    src = src.offset(1);
                }

                cmp = if dx < 0 { 0xffff_ffff } else { 0x0 };
                for j in xor_bits.iter().take(4) {
                    dx_mask = (cmp ^ j) as i32;
                    *src &= dx_mask;
                    src = src.offset(1);
                }
            }
        }
    }

    fn integral(&mut self) {
        let data = self.int_img.as_ptr();
        let len = (FeaturePool::K_NUM_INT_CHANNEL * self.width) as usize;

        unsafe {
            for r in 0..(self.height - 1) as isize {
                let row1 = data.offset(r * len as isize);
                let row2 = row1.offset(len as isize);
                math::vector_add(row1, row2, row2 as *mut i32, len);
            }

            for r in 0..self.height as isize {
                SurfMlpFeatureMap::vector_cumulative_add(
                    data.offset(r * len as isize),
                    len,
                    FeaturePool::K_NUM_INT_CHANNEL,
                );
            }
        }
    }

    #[inline]
    fn vector_cumulative_add(x: *const i32, len: usize, num_channel: u32) {
        SurfMlpFeatureMap::vector_cumulative_add_portable(x, len, num_channel);
    }

    fn vector_cumulative_add_portable(x: *const i32, len: usize, num_channel: u32) {
        unsafe {
            let num_channel = num_channel as usize;
            let cols = len / num_channel - 1;
            for i in 0..cols as isize {
                let col1 = x.offset(i * num_channel as isize);
                let col2 = col1.offset(num_channel as isize);
                math::vector_add(col1, col2, col2 as *mut i32, num_channel);
            }
        }
    }

    unsafe fn compute_feature_vector(&mut self, feature_id: usize, roi: Rectangle) {
        let feature = self.feature_pool.get_feature(feature_id);
        let feature_vec = self.feature_vectors[feature_id].as_mut_ptr();

        let init_cell_x = roi.x() + feature.patch.x();
        let init_cell_y = roi.y() + feature.patch.y();
        let k_num_int_channel = FeaturePool::K_NUM_INT_CHANNEL as isize;
        let cell_width: isize =
            (feature.patch.width() / feature.num_cell_per_row) as isize * k_num_int_channel;
        let cell_height: isize = (feature.patch.height() / feature.num_cell_per_col) as isize;
        let row_width: isize = (self.width as isize) * k_num_int_channel;

        let val = 0;
        let mut cell_top_left: Vec<*const i32> = vec![&val; k_num_int_channel as usize];
        let mut cell_top_right: Vec<*const i32> = vec![&val; k_num_int_channel as usize];
        let mut cell_bottom_left: Vec<*const i32> = vec![&val; k_num_int_channel as usize];
        let mut cell_bottom_right: Vec<*const i32> = vec![&val; k_num_int_channel as usize];

        let mut feature_value: *mut i32 = feature_vec;
        let int_img_ptr = self.int_img.as_ptr();
        let mut offset: isize;

        match (init_cell_x, init_cell_y) {
            (0, 0) => {
                offset = row_width * (cell_height - 1) + cell_width - k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_bottom_right[i] = int_img_ptr.offset(offset);
                    offset += 1;
                    *feature_value = *cell_bottom_right[i];
                    feature_value = feature_value.offset(1);
                    cell_top_right[i] = cell_bottom_right[i];
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *feature_value = *cell_bottom_right[j] - *cell_bottom_left[j];
                        feature_value = feature_value.offset(1);
                    }
                }
            }
            (_, 0) => {
                offset =
                    row_width * (cell_height - 1) + (init_cell_x - 1) as isize * k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_bottom_left[i] = int_img_ptr.offset(offset);
                    offset += 1;
                    cell_bottom_right[i] = cell_bottom_left[i].offset(cell_width);
                    *feature_value = *cell_bottom_right[i] - *cell_bottom_left[i];
                    feature_value = feature_value.offset(1);
                    cell_top_right[i] = cell_bottom_right[i];
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *feature_value = *cell_bottom_right[j] - *cell_bottom_left[j];
                        feature_value = feature_value.offset(1);
                    }
                }
            }
            (0, _) => {
                let mut tmp_cell_top_right: Vec<*const i32> =
                    vec![&val; k_num_int_channel as usize];

                offset = row_width * ((init_cell_y - 1) as isize) + cell_width - k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_top_right[i] = int_img_ptr.offset(offset);
                    offset += 1;
                    cell_bottom_right[i] = cell_top_right[i].offset(row_width * cell_height);
                    tmp_cell_top_right[i] = cell_bottom_right[i];
                    *feature_value = *cell_bottom_right[i] - *cell_top_right[i];
                    feature_value = feature_value.offset(1);
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_top_left[j] = cell_top_right[j];
                        cell_top_right[j] = cell_top_right[j].offset(cell_width);
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *feature_value = *cell_bottom_right[j] + *cell_top_left[j]
                            - *cell_top_right[j]
                            - *cell_bottom_left[j];
                        feature_value = feature_value.offset(1);
                    }
                }

                cell_top_right[..k_num_int_channel as usize]
                    .clone_from_slice(&tmp_cell_top_right[..k_num_int_channel as usize]);
            }
            (_, _) => {
                let mut tmp_cell_top_right: Vec<*const i32> =
                    vec![&val; k_num_int_channel as usize];

                offset = row_width * ((init_cell_y - 1) as isize)
                    + (init_cell_x - 1) as isize * k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_top_left[i] = int_img_ptr.offset(offset);
                    offset += 1;
                    cell_top_right[i] = cell_top_left[i].offset(cell_width);
                    cell_bottom_left[i] = cell_top_left[i].offset(row_width * cell_height);
                    cell_bottom_right[i] = cell_bottom_left[i].offset(cell_width);
                    *feature_value = *cell_bottom_right[i] + *cell_top_left[i]
                        - *cell_top_right[i]
                        - *cell_bottom_left[i];
                    feature_value = feature_value.offset(1);
                    tmp_cell_top_right[i] = cell_bottom_right[i];
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_top_left[j] = cell_top_right[j];
                        cell_top_right[j] = cell_top_right[j].offset(cell_width);
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *feature_value = *cell_bottom_right[j] + *cell_top_left[j]
                            - *cell_top_right[j]
                            - *cell_bottom_left[j];
                        feature_value = feature_value.offset(1);
                    }
                }

                cell_top_right[..k_num_int_channel as usize]
                    .clone_from_slice(&tmp_cell_top_right[..k_num_int_channel as usize]);
            }
        }

        offset = cell_height * row_width - feature.patch.width() as isize * k_num_int_channel
            + cell_width;
        for _ in 1..feature.num_cell_per_row {
            if init_cell_x == 0 {
                for j in 0..k_num_int_channel as usize {
                    cell_bottom_right[j] = cell_bottom_right[j].offset(offset);
                    *feature_value = *cell_bottom_right[j] - *cell_top_right[j];
                    feature_value = feature_value.offset(1);
                }
            } else {
                for j in 0..k_num_int_channel as usize {
                    cell_bottom_right[j] = cell_bottom_right[j].offset(offset);
                    cell_top_left[j] = cell_top_right[j].offset(-cell_width);
                    cell_bottom_left[j] = cell_bottom_right[j].offset(-cell_width);
                    *feature_value = *cell_bottom_right[j] + *cell_top_left[j]
                        - *cell_top_right[j]
                        - *cell_bottom_left[j];
                    feature_value = feature_value.offset(1);
                }
            }

            for _ in 1..feature.num_cell_per_row {
                for k in 0..k_num_int_channel as usize {
                    cell_top_left[k] = cell_top_right[k];
                    cell_top_right[k] = cell_top_right[k].offset(cell_width);
                    cell_bottom_left[k] = cell_bottom_right[k];
                    cell_bottom_right[k] = cell_bottom_right[k].offset(cell_width);
                    *feature_value = *cell_bottom_right[k] + *cell_top_left[k]
                        - *cell_bottom_left[k]
                        - *cell_top_right[k];
                    feature_value = feature_value.offset(1);
                }
            }

            for j in cell_top_right.iter_mut().take(k_num_int_channel as usize) {
                *j = j.offset(offset);
            }
        }
    }

    pub unsafe fn get_feature_vector(
        &mut self,
        feature_id: usize,
        feature_vec: *mut f32,
        roi: Rectangle,
    ) {
        self.compute_feature_vector(feature_id, roi);

        SurfMlpFeatureMap::normalize_feature_vector(
            &self.feature_vectors[feature_id],
            &mut self.feature_vectors_normalized[feature_id],
        );

        let feature_vec_normalized = self.feature_vectors_normalized[feature_id].as_ptr();
        let length = self.feature_vectors_normalized[feature_id].len();
        ptr::copy_nonoverlapping(feature_vec_normalized, feature_vec, length);
    }

    fn normalize_feature_vector(feature_vec: &[i32], feature_vec_normalized: &mut [f32]) {
        let prod: f64 = feature_vec
            .iter()
            .copied()
            .map(|value| f64::from(value * value))
            .sum();

        if prod != 0.0 {
            let norm = prod.sqrt() as f32;
            for (dst, src) in feature_vec_normalized.iter_mut().zip(feature_vec) {
                *dst = *src as f32 / norm;
            }
        } else {
            for dst in feature_vec_normalized {
                *dst = 0.0;
            }
        }
    }

    #[inline]
    pub fn get_feature_vector_dim(&self, feature_id: usize) -> usize {
        self.feature_pool.get_feature_vector_dim(feature_id)
    }
}

struct FeaturePool {
    sample_width: u32,
    sample_height: u32,
    patch_move_step_x: u32,
    patch_move_step_y: u32,
    patch_size_inc_step: u32,
    patch_min_width: u32,
    patch_min_height: u32,
    features: Vec<Feature>,
    patch_formats: Vec<PatchFormat>,
}

impl FeaturePool {
    const K_NUM_INT_CHANNEL: u32 = 8;

    #[inline]
    fn new() -> Self {
        FeaturePool {
            sample_width: 40,
            sample_height: 40,
            patch_move_step_x: 16,
            patch_move_step_y: 16,
            patch_size_inc_step: 1,
            patch_min_width: 16,
            patch_min_height: 16,
            features: Vec::new(),
            patch_formats: Vec::new(),
        }
    }

    fn add_patch_format(
        &mut self,
        width: u32,
        height: u32,
        num_cell_per_row: u32,
        num_cell_per_col: u32,
    ) {
        self.patch_formats.push(PatchFormat {
            width,
            height,
            num_cell_per_row,
            num_cell_per_col,
        });
    }

    fn create(&mut self) {
        let mut feature_vecs = Vec::new();

        if self.sample_height - self.patch_min_height <= self.sample_width - self.patch_min_width {
            for format in &self.patch_formats {
                for h in Seq::new(self.patch_min_height, |x| x + self.patch_size_inc_step)
                    .take_while(|x| *x <= self.sample_height)
                {
                    if h % format.num_cell_per_col != 0 || h % format.height != 0 {
                        continue;
                    }
                    let w = h / format.height * format.width;
                    if w % format.num_cell_per_row != 0
                        || w < self.patch_min_width
                        || w > self.sample_width
                    {
                        continue;
                    }
                    self.collect_features(
                        w,
                        h,
                        format.num_cell_per_row,
                        format.num_cell_per_col,
                        &mut feature_vecs,
                    );
                }
            }
        } else {
            for format in &self.patch_formats {
                // original condition was <= self.patch_min_width,
                // but it would not make sense to have a loop in such case
                for w in Seq::new(self.patch_min_width, |x| x + self.patch_size_inc_step)
                    .take_while(|x| *x <= self.sample_width)
                {
                    if w % format.num_cell_per_row != 0 || w % format.width != 0 {
                        continue;
                    }
                    let h = w / format.width * format.height;
                    if h % format.num_cell_per_col != 0
                        || h < self.patch_min_height
                        || h > self.sample_height
                    {
                        continue;
                    }
                    self.collect_features(
                        w,
                        h,
                        format.num_cell_per_row,
                        format.num_cell_per_col,
                        &mut feature_vecs,
                    );
                }
            }
        }

        self.features.append(&mut feature_vecs);
    }

    fn collect_features(
        &self,
        width: u32,
        height: u32,
        num_cell_per_row: u32,
        num_cell_per_col: u32,
        dest: &mut Vec<Feature>,
    ) {
        let y_lim = self.sample_height - height;
        let x_lim = self.sample_width - width;

        for y in Seq::new(0, |n| n + self.patch_move_step_y).take_while(|n| *n <= y_lim) {
            for x in Seq::new(0, |n| n + self.patch_move_step_x).take_while(|n| *n <= x_lim) {
                dest.push(Feature {
                    patch: Rectangle::new(x as i32, y as i32, width, height),
                    num_cell_per_row,
                    num_cell_per_col,
                });
            }
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.features.len()
    }

    #[inline]
    fn get_feature(&self, feature_id: usize) -> &Feature {
        &self.features[feature_id]
    }

    #[inline]
    fn get_feature_vector_dim(&self, feature_id: usize) -> usize {
        let feature = &self.features[feature_id];
        (feature.num_cell_per_col * feature.num_cell_per_row * FeaturePool::K_NUM_INT_CHANNEL)
            as usize
    }
}

struct PatchFormat {
    width: u32,
    height: u32,
    num_cell_per_row: u32,
    num_cell_per_col: u32,
}

struct Feature {
    patch: Rectangle,
    num_cell_per_row: u32,
    num_cell_per_col: u32,
}
