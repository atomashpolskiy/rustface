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

mod image_pyramid;

use std::mem;

pub use self::image_pyramid::{resize_image, ImageData, ImagePyramid};

#[derive(Clone, Debug)]
pub struct Rectangle {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

impl Rectangle {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Rectangle {
            x,
            y,
            width,
            height,
        }
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn set_x(&mut self, x: i32) {
        self.x = x;
    }

    pub fn y(&self) -> i32 {
        self.y
    }

    pub fn set_y(&mut self, y: i32) {
        self.y = y;
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn set_width(&mut self, width: u32) {
        self.width = width;
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn set_height(&mut self, height: u32) {
        self.height = height;
    }
}

#[derive(Clone, Debug)]
pub struct FaceInfo {
    bbox: Rectangle,
    roll: f64,
    pitch: f64,
    yaw: f64,
    score: f64,
}

impl Default for FaceInfo {
    fn default() -> Self {
        FaceInfo {
            bbox: Rectangle::new(0, 0, 0, 0),
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            score: 0.0,
        }
    }
}

impl FaceInfo {
    pub fn new() -> Self {
        FaceInfo::default()
    }

    pub fn bbox(&self) -> &Rectangle {
        &self.bbox
    }

    pub fn bbox_mut(&mut self) -> &mut Rectangle {
        &mut self.bbox
    }

    pub fn set_score(&mut self, score: f64) {
        self.score = score;
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

pub struct Seq<T, G>
where
    G: Fn(&T) -> T + Sized,
{
    generator: G,
    next: T,
}

impl<T, G> Seq<T, G>
where
    G: Fn(&T) -> T + Sized,
{
    pub fn new(first_element: T, generator: G) -> Self {
        Seq {
            generator,
            next: first_element,
        }
    }
}

impl<T, G> Iterator for Seq<T, G>
where
    G: Fn(&T) -> T + Sized,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = (self.generator)(&self.next);
        let current = mem::replace(&mut self.next, next);
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::Seq;

    #[test]
    pub fn test_seq_take() {
        let seq = Seq::new(0, |x| x + 1);
        assert_eq!(vec![0, 1, 2, 3, 4], seq.take(5).collect::<Vec<i32>>());
    }

    #[test]
    pub fn test_seq_take_while() {
        let seq = Seq::new(0, |x| x + 1);
        assert_eq!(
            vec![0, 1, 2, 3, 4],
            seq.take_while(|x| *x < 5).collect::<Vec<i32>>()
        );
    }
}
