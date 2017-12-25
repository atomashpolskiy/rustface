extern crate byteorder;
extern crate num;

mod common;
mod math;
mod feat;
mod classifier;
pub mod model;

pub use common::ImageData;
pub use common::FaceInfo;

use std::{cmp, io, ptr};
use std::cmp::Ordering::*;
use std::cell::RefCell;
use std::rc::Rc;
use common::{ImagePyramid, Rectangle, Seq};
use model::Model;

pub fn create_detector(path_to_model: &str) -> Result<Box<Detector>, io::Error> {
    let model = model::load_model(path_to_model)?;
    Ok(Box::new(FuStDetector::new(model)))
}

pub trait Detector {
    fn detect(&mut self, image: &mut ImageData) -> Vec<FaceInfo>;
    fn set_window_size(&mut self, wnd_size: u32);
    fn set_slide_window_step(&mut self, step_x: u32, step_y: u32);
    fn set_min_face_size(&mut self, min_face_size: u32);
    fn set_max_face_size(&mut self, max_face_size: u32);
    fn set_pyramid_scale_factor(&mut self, scale_factor: f32);
    fn set_score_thresh(&mut self, thresh: f64);
}

impl Detector for FuStDetector {
    fn detect(&mut self, image: &mut ImageData) -> Vec<FaceInfo> {
        if !is_legal_image(image) {
            panic!("Illegal image");
        }

        let mut min_img_size = cmp::min(image.height(), image.width());
        if self.max_face_size > 0 {
            min_img_size = cmp::min(self.max_face_size as u32, min_img_size);
        }

        const K_WND_SIZE: f32 = 40.0;

        let mut image_pyramid = ImagePyramid::new();
        image_pyramid.set_image_1x(image.data(), image.width(), image.height());
        image_pyramid.set_max_scale(K_WND_SIZE / self.min_face_size as f32);
        image_pyramid.set_min_scale(K_WND_SIZE / min_img_size as f32);
        image_pyramid.set_scale_step(self.image_pyramid_scale_factor);
        self.set_window_size(K_WND_SIZE as u32);

        self.detect_impl(&mut image_pyramid).into_iter()
            .filter(|x| x.score() >= self.cls_thresh)
            .collect()
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

    fn set_min_face_size(&mut self, min_face_size: u32) {
        if min_face_size < 20 {
            panic!("Illegal min face size");
        }
        self.min_face_size = min_face_size as i32;
    }

    fn set_max_face_size(&mut self, max_face_size: u32) {
        self.max_face_size = max_face_size as i32;
    }

    fn set_pyramid_scale_factor(&mut self, scale_factor: f32) {
        if scale_factor < 0.01 || scale_factor > 0.99 {
            panic!("Illegal scale factor");
        }
        self.image_pyramid_scale_factor = scale_factor;
    }

    fn set_score_thresh(&mut self, thresh: f64) {
        if thresh <= 0.0 {
            panic!("Illegal value");
        }
        self.cls_thresh = thresh;
    }
}

fn is_legal_image(image: &ImageData) -> bool {
    image.num_channels() == 1 && image.width() > 0 && image.height() > 0
}

struct FuStDetector {
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

impl FuStDetector {
    fn new(model: Model) -> Self {
        let wnd_size = 40;
        let slide_wnd_step_x = 4;
        let slide_wnd_step_y = 4;

        FuStDetector {
            model,
            wnd_data_buf: Vec::with_capacity((wnd_size * wnd_size) as usize),
            wnd_data: Vec::with_capacity((wnd_size * wnd_size) as usize),
            wnd_size,
            slide_wnd_step_x,
            slide_wnd_step_y,
            min_face_size: 20,
            max_face_size: -1,
            cls_thresh: 3.85,
            image_pyramid_scale_factor: 0.8,
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
                for _y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::copy_nonoverlapping(src, dest, len);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                    }
                }
            },
            (0, _) => {
                for _y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::copy_nonoverlapping(src, dest, len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                        ptr::write_bytes(dest.offset(-pad_right as isize), 0, pad_right as usize);
                    }
                }
            },
            (_, 0) => {
                for _y in pad_top..(roi_height - pad_bottom) {
                    unsafe {
                        ptr::write_bytes(dest, 0, pad_left as usize);
                        ptr::copy_nonoverlapping(src, dest.offset(pad_left as isize), len2);
                        src = src.offset(img_width as isize);
                        dest = dest.offset(roi_width as isize);
                    }
                }
            },
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

    fn detect_impl(&mut self, image: &mut ImagePyramid) -> Vec<FaceInfo> {
        let mut scale_factor = 0.0;
        let mut image_scaled_optional = image.get_next_scale_image(&mut scale_factor);

        let wnd_info = Rc::new(RefCell::new(FaceInfo::new()));
        let first_hierarchy_size = self.model.get_hierarchy_size(0) as usize;
        let mut proposals: Vec<Rc<RefCell<Vec<Rc<RefCell<FaceInfo>>>>>> = vec![Rc::new(RefCell::new(vec![])); first_hierarchy_size];
        let mut proposals_nms: Vec<Rc<RefCell<Vec<Rc<RefCell<FaceInfo>>>>>> = vec![Rc::new(RefCell::new(vec![])); first_hierarchy_size];

        loop {
            match image_scaled_optional {
                Some(ref image_scaled) => {
                    self.model.get_classifiers()[0].compute(image_scaled);

                    let width = (self.wnd_size as f32 / scale_factor + 0.5) as u32;
                    wnd_info.borrow_mut().bbox_mut().set_width(width);
                    wnd_info.borrow_mut().bbox_mut().set_height(width);

                    let step_x = self.slide_wnd_step_x;
                    let step_y = self.slide_wnd_step_y;
                    let max_x = image_scaled.width() - self.wnd_size;
                    let max_y = image_scaled.height() - self.wnd_size;

                    for y in Seq::new(0, move |n| n + step_y).take_while(move |n| *n <= max_y) {
                        for x in Seq::new(0, move |n| n + step_x).take_while(move |n| *n <= max_x) {

                            self.model.get_classifiers()[0].set_roi(Rectangle::new(x as i32, y as i32, self.wnd_size, self.wnd_size));

                            wnd_info.borrow_mut().bbox_mut().set_x((x as f32 / scale_factor + 0.5) as i32);
                            wnd_info.borrow_mut().bbox_mut().set_y((y as f32 / scale_factor + 0.5) as i32);

                            for i in 0..first_hierarchy_size {
                                let score = (&mut *self.model.get_classifiers()[i]).classify(None);
                                if score.is_positive() {
                                    wnd_info.borrow_mut().set_score(score.score() as f64);
                                    proposals[i].borrow_mut().push(Rc::new(RefCell::new(wnd_info.borrow().clone())));
                                }
                            }
                        }
                    }
                },
                None => break,
            }
            image_scaled_optional = image.get_next_scale_image(&mut scale_factor);
        }

        for i in 0..first_hierarchy_size {
            non_maximum_suppression(proposals[i].borrow_mut().as_mut(), proposals_nms[i].borrow_mut().as_mut(), 0.8);
            proposals[i].borrow_mut().clear();
        }

        let image1x = image.get_image_1x();
        let mut mlp_predicts: Vec<f32> = Vec::with_capacity(4);

        let mut cls_idx = first_hierarchy_size;
        let mut model_idx = first_hierarchy_size;
        let mut buf_idx = vec![];

        for i in 1..self.model.get_hierarchy_count() {

            let hierarchy_size_i = self.model.get_hierarchy_size(i) as usize;
            buf_idx.resize(hierarchy_size_i, 0);

            for j in 0..hierarchy_size_i as usize {
                let r;
                {
                    let wnd_src = self.model.get_wnd_src(cls_idx);
                    let num_wnd_src = wnd_src.len();
                    buf_idx[j] = wnd_src[0];
                    r = buf_idx[j] as usize;
                    proposals[r].borrow_mut().clear();

                    for k in 0..num_wnd_src {
                        for ref item in proposals_nms[wnd_src[k] as usize].borrow().iter() {
                            let last_index = proposals[r].borrow().len() - 1;
                            proposals[r].borrow_mut().insert(last_index, Rc::clone(item));
                        }
                    }
                }

                let k_max = self.model.get_num_stage(cls_idx);
                for k in 0..k_max {
                    let mut bbox_id = 0;
                    {
                        let num_wnd = proposals[r].borrow().len();
                        let bboxes = &proposals[r].borrow_mut();


                        for m in 0..num_wnd {
                            if bboxes[m].borrow().bbox().x() + bboxes[m].borrow().bbox().width() as i32 <= 0 ||
                                bboxes[m].borrow().bbox().y() + bboxes[m].borrow().bbox().height() as i32 <= 0 {
                                continue;
                            }

                            self.get_window_data(&image1x, bboxes[m].borrow_mut().bbox_mut());
                            let img_temp = ImageData::new(self.wnd_data.as_ptr(), self.wnd_size, self.wnd_size);
                            self.model.get_classifiers()[model_idx].compute(&img_temp);
                            self.model.get_classifiers()[model_idx].set_roi(Rectangle::new(0, 0, self.wnd_size, self.wnd_size));

                            let new_score = self.model.get_classifiers()[model_idx].classify(Some(&mut mlp_predicts));
                            if new_score.is_positive() {
                                let x = bboxes[m].borrow().bbox().x() as f32;
                                let y = bboxes[m].borrow().bbox().y() as f32;
                                let w = bboxes[m].borrow().bbox().width() as f32;
                                let h = bboxes[m].borrow().bbox().height() as f32;

                                let bbox_w = ((mlp_predicts[3] * 2.0 - 1.0) * w + w + 0.5) as f32;
                                bboxes[bbox_id].borrow_mut().bbox_mut().set_width(bbox_w as u32);
                                bboxes[bbox_id].borrow_mut().bbox_mut().set_height(bbox_w as u32);

                                bboxes[bbox_id].borrow_mut().bbox_mut().set_x(
                                    ((mlp_predicts[1] * 2.0 - 1.0) * w + x + (w - bbox_w) * 0.5 + 0.5) as i32);

                                bboxes[bbox_id].borrow_mut().bbox_mut().set_y(
                                    ((mlp_predicts[2] * 2.0 - 1.0) * h + y + (h - bbox_w) * 0.5 + 0.5) as i32);

                                bboxes[bbox_id].borrow_mut().set_score(new_score.score() as f64);

                                bbox_id += 1;
                            }
                        }
                    }

                    proposals[r].borrow_mut().truncate(bbox_id);

                    if proposals[r].as_ptr() != proposals_nms[r].as_ptr() {
                        if k < (k_max - 1) {
                            non_maximum_suppression(proposals[r].borrow_mut().as_mut(), proposals_nms[r].borrow_mut().as_mut(), 0.8);
                            proposals[r] = Rc::clone(&proposals_nms[r]);
                        } else if i == (self.model.get_hierarchy_count() - 1) {
                            non_maximum_suppression(proposals[r].borrow_mut().as_mut(), proposals_nms[r].borrow_mut().as_mut(), 0.3);
                            proposals[r] = Rc::clone(&proposals_nms[r]);
                        }
                    }

                    model_idx += 1;
                }

                cls_idx += 1;
            }

            for j in 0..hierarchy_size_i {
                proposals_nms[j] = Rc::clone(&proposals[j]);
            }
        }

        // drop duplicate refs..
        drop(proposals);
        proposals_nms.truncate(1);

        proposals_nms.into_iter().take(1)
            .map(|rc| Rc::try_unwrap(rc).and_then(|cell| Ok(cell.into_inner())).ok().unwrap())
            .into_iter()
            .flat_map(|vec| vec.into_iter())
            .map(|rc| Rc::try_unwrap(rc).and_then(|cell| Ok(cell.into_inner())).ok().unwrap())
            .collect()
    }
}

fn non_maximum_suppression(bboxes: &mut Vec<Rc<RefCell<FaceInfo>>>, bboxes_nms: &mut Vec<Rc<RefCell<FaceInfo>>>, iou_thresh: f32) {
    bboxes_nms.clear();
    bboxes.sort_by(|x, y| {
        let x_score = x.borrow().score();
        let y_score = y.borrow().score();
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

        bboxes_nms.push(Rc::clone(&bboxes[select_idx]));
        mask_merged[select_idx] = true;

        let mut score;
        {
            score = bboxes_nms.last().unwrap().borrow().score();
        }

        let area1;
        let x1;
        let y1;
        let x2;
        let y2;
        {
            area1 = (bboxes[select_idx].borrow().bbox().width() * bboxes[select_idx].borrow().bbox().height()) as f32;
            x1 = bboxes[select_idx].borrow().bbox().x();
            y1 = bboxes[select_idx].borrow().bbox().y();
            x2 = bboxes[select_idx].borrow().bbox().x() + bboxes[select_idx].borrow().bbox().width() as i32 - 1;
            y2 = bboxes[select_idx].borrow().bbox().y() + bboxes[select_idx].borrow().bbox().height() as i32 - 1;
        }

        select_idx += 1;

        for i in select_idx..bboxes.len() {
            if mask_merged[i] {
                continue;
            }

            let x = cmp::max(x1, bboxes[i].borrow().bbox().x());
            let y = cmp::max(y1, bboxes[i].borrow().bbox().y());
            let w = cmp::min(x2, (bboxes[i].borrow().bbox().x() + bboxes[i].borrow().bbox().width() as i32 - 1)) - x + 1;
            let h = cmp::min(y2, (bboxes[i].borrow().bbox().y() + bboxes[i].borrow().bbox().height() as i32 - 1)) - y + 1;

            if w <= 0 || h <= 0 {
                continue;
            }

            let area2 = (bboxes[i].borrow().bbox().width() * bboxes[i].borrow().bbox().height()) as f32;
            let area_intersect = (w * h) as f32;
            let area_union = area1 + area2 - area_intersect;
            if area_intersect / area_union > iou_thresh {
                mask_merged[i] = true;
                let bbox_i_score = bboxes[i].borrow().score();
                score += bbox_i_score;

            }
        }

        bboxes_nms.last().unwrap().borrow_mut().set_score(score);
    }
}