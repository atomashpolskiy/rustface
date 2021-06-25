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

use std::cmp::Ordering::*;
use std::{cmp, ptr};

use crate::classifier::{Classifier, Score, SurfMlpBuffers};
use crate::common::{resize_image, FaceInfo, ImageData, ImagePyramid, Rectangle, Seq};
use crate::feat::FeatureMap;
use crate::feat::LabBoostedFeatureMap;
use crate::feat::SurfMlpFeatureMap;
use crate::model::Model;
use crate::Detector;

const FUST_MIN_WINDOW_SIZE: u32 = 20;

impl Detector for FuStDetector {
    fn detect(&mut self, image: &ImageData) -> Vec<FaceInfo> {
        if !is_legal_image(image) {
            panic!("Illegal image: {:?}", image);
        }

        let mut min_img_size = cmp::min(image.height(), image.width());
        if self.max_face_size > 0 {
            min_img_size = cmp::min(self.max_face_size as u32, min_img_size);
        }

        const K_WND_SIZE: f32 = 40.0;

        let mut image_pyramid = ImagePyramid::new();
        image_pyramid.set_image_1x(image.data(), image.width(), image.height());
        // TODO: uncomment (expect perf hit)
        //        image_pyramid.set_max_scale(K_WND_SIZE / self.min_face_size as f32);
        image_pyramid.set_min_scale(K_WND_SIZE / min_img_size as f32);
        image_pyramid.set_scale_step(self.image_pyramid_scale_factor);
        self.set_window_size(K_WND_SIZE as u32);

        self.detect_impl(&mut image_pyramid)
            .into_iter()
            .filter(|x| x.score() >= self.cls_thresh)
            .collect()
    }

    fn set_window_size(&mut self, wnd_size: u32) {
        if wnd_size < FUST_MIN_WINDOW_SIZE {
            panic!("Illegal window size: {}", wnd_size);
        }
        self.wnd_size = wnd_size;
    }

    fn set_slide_window_step(&mut self, step_x: u32, step_y: u32) {
        if step_x <= 0 {
            panic!("Illegal horizontal step: {}", step_x);
        }
        if step_y <= 0 {
            panic!("Illegal vertical step: {}", step_y);
        }
        self.slide_wnd_step_x = step_x;
        self.slide_wnd_step_y = step_y;
    }

    fn set_min_face_size(&mut self, min_face_size: u32) {
        if min_face_size < FUST_MIN_WINDOW_SIZE {
            panic!("Illegal min face size: {}", min_face_size);
        }
        self.min_face_size = min_face_size as i32;
    }

    #[inline]
    fn set_max_face_size(&mut self, max_face_size: u32) {
        self.max_face_size = max_face_size as i32;
    }

    fn set_pyramid_scale_factor(&mut self, scale_factor: f32) {
        if scale_factor < 0.01 || scale_factor > 0.99 {
            panic!("Illegal scale factor: {}", scale_factor);
        }
        self.image_pyramid_scale_factor = scale_factor;
    }

    fn set_score_thresh(&mut self, thresh: f64) {
        if thresh <= 0.0 {
            panic!("Illegal threshold: {}", thresh);
        }
        self.cls_thresh = thresh;
    }
}

#[inline]
fn is_legal_image(image: &ImageData) -> bool {
    image.num_channels() == 1 && image.width() > 0 && image.height() > 0
}

pub struct FuStDetector {
    feature_maps: FeatureMaps,
    model: Model,
    wnd_data_buf: Vec<u8>,
    wnd_data: Vec<u8>,
    wnd_size: u32,
    slide_wnd_step_x: u32,
    slide_wnd_step_y: u32,
    min_face_size: i32,
    max_face_size: i32,
    cls_thresh: f64,
    image_pyramid_scale_factor: f32,
}

struct FeatureMaps {
    lab_boosted: LabBoostedFeatureMap,
    surf_mlp: SurfMlpFeatureMap,
    surf_buf: SurfMlpBuffers,
}

impl FuStDetector {
    pub fn new(model: Model) -> Self {
        let wnd_size = 40;
        let slide_wnd_step_x = 4;
        let slide_wnd_step_y = 4;

        FuStDetector {
            feature_maps: FeatureMaps {
                lab_boosted: LabBoostedFeatureMap::new(),
                surf_mlp: SurfMlpFeatureMap::new(),
                surf_buf: SurfMlpBuffers::new(),
            },
            model,
            wnd_data_buf: vec![0; (wnd_size * wnd_size) as usize],
            wnd_data: vec![0; (wnd_size * wnd_size) as usize],
            wnd_size,
            slide_wnd_step_x,
            slide_wnd_step_y,
            min_face_size: 20,
            max_face_size: -1,
            cls_thresh: 3.85,
            image_pyramid_scale_factor: 0.8,
        }
    }

    fn feature_map_for_classifier<'a>(
        classifier: &Classifier,
        maps: &'a mut FeatureMaps,
    ) -> &'a mut dyn FeatureMap {
        match classifier {
            Classifier::LabBoosted(_) => &mut maps.lab_boosted,
            Classifier::SurfMlp(_) => &mut maps.surf_mlp,
        }
    }

    fn classify_with_classifier(
        classifier: &Classifier,
        output: Option<&mut Vec<f32>>,
        maps: &mut FeatureMaps,
        roi: Rectangle,
    ) -> Score {
        match classifier {
            Classifier::SurfMlp(c) => {
                c.classify(output, &mut maps.surf_buf, &mut maps.surf_mlp, roi)
            }
            Classifier::LabBoosted(c) => c.classify(&mut maps.lab_boosted, roi),
        }
    }

    fn get_window_data(&mut self, img: &ImageData, wnd: &mut Rectangle) {
        let roi = wnd;

        let roi_width = roi.width() as i32;
        let roi_height = roi.height() as i32;
        let img_width = img.width() as i32;
        let img_height = img.height() as i32;

        let pad_right = cmp::max(roi.x() + roi_width - img_width, 0);
        let pad_left = if roi.x() >= 0 {
            0
        } else {
            let t = roi.x();
            roi.set_x(0);
            -t
        };
        let pad_bottom = cmp::max(roi.y() + roi_height - img_height, 0);
        let pad_top = if roi.y() >= 0 {
            0
        } else {
            let t = roi.y();
            roi.set_y(0);
            -t
        };

        self.wnd_data_buf
            .resize((roi_width * roi_height) as usize, 0);
        let mut src;
        unsafe {
            src = img
                .data()
                .as_ptr()
                .offset((roi.y() * img_width + roi.x()) as isize);
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
                for _y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::copy_nonoverlapping(src, dest, len);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                    }
                }
            }
            (0, _) => {
                for _y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::copy_nonoverlapping(src, dest, len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                        ptr::write_bytes(dest.offset(-pad_right as isize), 0, pad_right as usize);
                    }
                }
            }
            (_, 0) => {
                for _y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::write_bytes(dest, 0, pad_left as usize);
                        ptr::copy_nonoverlapping(src, dest.offset(pad_left as isize), len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                    }
                }
            }
            (_, _) => {
                for _y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::write_bytes(dest, 0, pad_left as usize);
                        ptr::copy_nonoverlapping(src, dest.offset(pad_left as isize), len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                        ptr::write_bytes(dest.offset(-pad_right as isize), 0, pad_right as usize);
                    }
                }
            }
        }

        if pad_bottom > 0 {
            unsafe {
                ptr::write_bytes(dest, 0, len * pad_bottom as usize);
            }
        }

        let src_img = ImageData::new(&self.wnd_data_buf, roi.width(), roi.height());
        resize_image(&src_img, &mut self.wnd_data, self.wnd_size, self.wnd_size);
    }

    fn detect_impl(&mut self, image: &mut ImagePyramid) -> Vec<FaceInfo> {
        let first_hierarchy_size = self.model.get_hierarchy_size(0) as usize;
        let mut proposals = vec![Vec::new(); first_hierarchy_size];
        let mut proposals_nms = vec![Vec::new(); first_hierarchy_size];

        while let Some((image_scaled, scale_factor)) = image.get_next_scale_image() {
            Self::feature_map_for_classifier(
                &self.model.get_classifiers()[0],
                &mut self.feature_maps,
            )
            .compute(&image_scaled);

            let step_x = self.slide_wnd_step_x;
            let step_y = self.slide_wnd_step_y;
            let max_x = image_scaled.width() - self.wnd_size;
            let max_y = image_scaled.height() - self.wnd_size;

            for y in Seq::new(0, move |n| n + step_y).take_while(move |n| *n <= max_y) {
                for x in Seq::new(0, move |n| n + step_x).take_while(move |n| *n <= max_x) {
                    let rect = Rectangle::new(x as i32, y as i32, self.wnd_size, self.wnd_size);
                    Self::feature_map_for_classifier(
                        &self.model.get_classifiers()[0],
                        &mut self.feature_maps,
                    );

                    for (classifier, proposal) in self
                        .model
                        .get_classifiers()
                        .iter()
                        .zip(proposals.iter_mut())
                        .take(first_hierarchy_size)
                    {
                        let score = Self::classify_with_classifier(
                            classifier,
                            None,
                            &mut self.feature_maps,
                            rect,
                        );
                        if score.is_positive() {
                            let mut wnd_info = FaceInfo::new();
                            let bbox = wnd_info.bbox_mut();
                            bbox.set_x((x as f32 / scale_factor + 0.5) as i32);
                            bbox.set_y((y as f32 / scale_factor + 0.5) as i32);
                            let width = (self.wnd_size as f32 / scale_factor + 0.5) as u32;
                            bbox.set_width(width);
                            bbox.set_height(width);
                            wnd_info.set_score(f64::from(score.score()));
                            proposal.push(wnd_info);
                        }
                    }
                }
            }
        }

        for i in 0..first_hierarchy_size {
            non_maximum_suppression(&mut proposals[i], &mut proposals_nms[i], 0.8);
            proposals[i].clear();
        }

        let image1x = image.get_image_1x();
        let mut mlp_predicts: Vec<f32> = vec![0.0; 4];

        let mut cls_idx = first_hierarchy_size;
        let mut model_idx = first_hierarchy_size;
        let mut buf_idx: Vec<i32> = Vec::new();

        for i in 1..self.model.get_hierarchy_count() {
            let hierarchy_size_i = self.model.get_hierarchy_size(i) as usize;
            if buf_idx.len() < hierarchy_size_i {
                buf_idx.resize(hierarchy_size_i, 0);
            }

            for r in buf_idx.iter_mut().take(hierarchy_size_i as usize) {
                {
                    let wnd_src = self.model.get_wnd_src(cls_idx);
                    *r = wnd_src[0];
                    let num_wnd_src = wnd_src.len();
                    let r = *r as usize;
                    proposals[r].clear();

                    for k in wnd_src.iter().take(num_wnd_src) {
                        for item in &proposals_nms[*k as usize] {
                            let last_index = proposals[r].len();
                            proposals[r].insert(last_index, item.clone());
                        }
                    }
                }
                let r = *r as usize;

                let k_max = self.model.get_num_stage(cls_idx);
                for k in 0..k_max {
                    let mut bbox_id = 0;
                    {
                        let num_wnd = proposals[r].len();
                        let bboxes = &mut proposals[r];

                        for m in 0..num_wnd {
                            if bboxes[m].bbox().x() + bboxes[m].bbox().width() as i32 <= 0
                                || bboxes[m].bbox().y() + bboxes[m].bbox().height() as i32 <= 0
                            {
                                continue;
                            }

                            self.get_window_data(&image1x, bboxes[m].bbox_mut());
                            let img_temp =
                                ImageData::new(&self.wnd_data, self.wnd_size, self.wnd_size);
                            let classifier = &self.model.get_classifiers()[model_idx];
                            Self::feature_map_for_classifier(classifier, &mut self.feature_maps)
                                .compute(&img_temp);
                            let rect = Rectangle::new(0, 0, self.wnd_size, self.wnd_size);

                            let new_score = Self::classify_with_classifier(
                                classifier,
                                Some(&mut mlp_predicts),
                                &mut self.feature_maps,
                                rect,
                            );
                            if new_score.is_positive() {
                                let x = bboxes[m].bbox().x() as f32;
                                let y = bboxes[m].bbox().y() as f32;
                                let w = bboxes[m].bbox().width() as f32;
                                let h = bboxes[m].bbox().height() as f32;

                                let bbox_w = ((mlp_predicts[3] * 2.0 - 1.0) * w + w + 0.5).floor();
                                bboxes[bbox_id].bbox_mut().set_width(bbox_w as u32);
                                bboxes[bbox_id].bbox_mut().set_height(bbox_w as u32);

                                bboxes[bbox_id].bbox_mut().set_x(
                                    ((mlp_predicts[1] * 2.0 - 1.0) * w
                                        + x
                                        + (w - bbox_w) * 0.5
                                        + 0.5)
                                        .floor() as i32,
                                );

                                bboxes[bbox_id].bbox_mut().set_y(
                                    ((mlp_predicts[2] * 2.0 - 1.0) * h
                                        + y
                                        + (h - bbox_w) * 0.5
                                        + 0.5)
                                        .floor() as i32,
                                );

                                bboxes[bbox_id].set_score(f64::from(new_score.score()));

                                bbox_id += 1;
                            }
                        }
                    }

                    proposals[r].truncate(bbox_id);

                    if k < (k_max - 1) {
                        non_maximum_suppression(&mut proposals[r], &mut proposals_nms[r], 0.8);
                        proposals[r] = proposals_nms[r].clone();
                    } else if i == (self.model.get_hierarchy_count() - 1) {
                        non_maximum_suppression(&mut proposals[r], &mut proposals_nms[r], 0.3);
                        proposals[r] = proposals_nms[r].clone();
                    }

                    model_idx += 1;
                }

                cls_idx += 1;
            }

            for j in 0..hierarchy_size_i {
                proposals_nms[j] = proposals[buf_idx[j] as usize].clone();
            }
        }

        proposals_nms[0].clone()
    }
}

fn non_maximum_suppression(
    bboxes: &mut Vec<FaceInfo>,
    bboxes_nms: &mut Vec<FaceInfo>,
    iou_thresh: f32,
) {
    bboxes_nms.clear();
    bboxes.sort_by(|x, y| {
        let x_score = x.score();
        let y_score = y.score();
        if x_score > y_score {
            // x goes before y
            Less
        } else if x_score < y_score {
            Greater
        } else {
            Equal
        }
    });

    let mut select_idx = 0;
    let mut mask_merged = vec![false; bboxes.len()];

    loop {
        while select_idx < bboxes.len() && mask_merged[select_idx] {
            select_idx += 1;
        }

        if select_idx == bboxes.len() {
            break;
        }

        bboxes_nms.push(bboxes[select_idx].clone());
        mask_merged[select_idx] = true;

        let mut score;
        {
            score = bboxes_nms.last().unwrap().score();
        }

        let area1;
        let x1;
        let y1;
        let x2;
        let y2;
        {
            area1 = (bboxes[select_idx].bbox().width() * bboxes[select_idx].bbox().height()) as f32;
            x1 = bboxes[select_idx].bbox().x();
            y1 = bboxes[select_idx].bbox().y();
            x2 = bboxes[select_idx].bbox().x() + bboxes[select_idx].bbox().width() as i32 - 1;
            y2 = bboxes[select_idx].bbox().y() + bboxes[select_idx].bbox().height() as i32 - 1;
        }

        select_idx += 1;

        for i in select_idx..bboxes.len() {
            if mask_merged[i] {
                continue;
            }

            let x = cmp::max(x1, bboxes[i].bbox().x());
            let y = cmp::max(y1, bboxes[i].bbox().y());
            let w = cmp::min(
                x2,
                bboxes[i].bbox().x() + bboxes[i].bbox().width() as i32 - 1,
            ) - x
                + 1;
            let h = cmp::min(
                y2,
                bboxes[i].bbox().y() + bboxes[i].bbox().height() as i32 - 1,
            ) - y
                + 1;

            if w <= 0 || h <= 0 {
                continue;
            }

            let area2 = (bboxes[i].bbox().width() * bboxes[i].bbox().height()) as f32;
            let area_intersect = (w * h) as f32;
            let area_union = area1 + area2 - area_intersect;
            if area_intersect / area_union > iou_thresh {
                mask_merged[i] = true;
                let bbox_i_score = bboxes[i].score();
                score += bbox_i_score;
            }
        }

        bboxes_nms.last_mut().unwrap().set_score(score);
    }
}
