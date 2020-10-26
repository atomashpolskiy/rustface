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

use std::rc::Rc;

use crate::math;

use super::Score;
use crate::common::{ImageData, Rectangle};
use crate::feat::{FeatureMap, SurfMlpFeatureMap};
use std::cell::RefCell;
use std::ptr;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

struct TwoWayBuffer {
    input: Vec<f32>,
    output: Vec<f32>,
}

impl TwoWayBuffer {
    fn new() -> Self {
        TwoWayBuffer {
            input: vec![],
            output: vec![],
        }
    }

    fn get_buffers(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>) {
        (&mut self.input, &mut self.output)
    }

    fn get_input(&mut self) -> &mut Vec<f32> {
        &mut self.input
    }

    fn swap(&mut self) {
        unsafe {
            ptr::swap(&mut self.input, &mut self.output);
        }
    }
}

pub struct SurfMlpClassifier {
    feature_map: Rc<RefCell<SurfMlpFeatureMap>>,
    feature_ids: Vec<i32>,
    thresh: f32,
    layers: Vec<Layer>,
    layers_buf: TwoWayBuffer,
    input_buf: Option<Vec<f32>>,
    output_buf: Option<Vec<f32>>,
}

impl SurfMlpClassifier {
    pub fn new(feature_map: Rc<RefCell<SurfMlpFeatureMap>>) -> Self {
        SurfMlpClassifier {
            feature_map,
            feature_ids: vec![],
            thresh: 0.0,
            layers: vec![],
            layers_buf: TwoWayBuffer::new(),
            input_buf: None,
            output_buf: None,
        }
    }

    pub fn add_feature_id(&mut self, feature_id: i32) {
        self.feature_ids.push(feature_id);
    }

    pub fn set_threshold(&mut self, thresh: f32) {
        self.thresh = thresh;
    }

    pub fn add_layer(
        &mut self,
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    ) {
        self.layers.push(Layer {
            input_dim,
            output_dim,
            weights,
            biases,
            act_func: Box::new(Self::relu),
        })
    }

    pub fn add_output_layer(
        &mut self,
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    ) {
        self.layers.push(Layer {
            input_dim,
            output_dim,
            weights,
            biases,
            act_func: Box::new(Self::sigmoid),
        })
    }

    fn relu(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn compute_internal(&mut self) {
        let input = self.input_buf.as_ref().unwrap();
        let output = self.output_buf.as_mut().unwrap();

        self.layers_buf
            .get_input()
            .resize(self.layers[0].output_size(), 0.0);
        self.layers[0].compute(input, self.layers_buf.get_input());

        for i in 1..(self.layers.len() - 1) {
            {
                let layer = &self.layers[i];
                let (input_buf, output_buf) = self.layers_buf.get_buffers();
                output_buf.resize(layer.output_size(), 0.0);
                layer.compute(input_buf, output_buf);
            }
            self.layers_buf.swap();
        }

        let last_layer = &self.layers[self.layers.len() - 1];
        last_layer.compute(self.layers_buf.get_input(), output);
    }
}

type ActFunc = dyn Fn(f32) -> f32 + Sync;

struct Layer {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    act_func: Box<ActFunc>,
}

impl Layer {
    fn compute(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "rayon")]
        let it = self.weights.par_chunks(self.input_dim);

        #[cfg(not(feature = "rayon"))]
        let it = self.weights.chunks(self.input_dim);

        it
            .zip(&self.biases)
            .zip(output)
            .for_each(|((weights, bias), output)| {
                let x = math::vector_inner_product(input, weights) + bias;
                *output = (self.act_func)(x);
            });
    }

    fn input_size(&self) -> usize {
        self.input_dim
    }

    fn output_size(&self) -> usize {
        self.output_dim
    }
}

impl SurfMlpClassifier {
    pub fn classify(&mut self, output: Option<&mut Vec<f32>>) -> Score {
        if self.input_buf.is_none() {
            let input_layer = self.layers.get(0).expect("No layers");
            self.input_buf = Some(vec![0.0; input_layer.input_size()]);
        }
        if self.output_buf.is_none() {
            let num_layers = self.layers.len();
            let output_layer = self.layers.get(num_layers - 1).expect("No layers");
            self.output_buf = Some(vec![0.0; output_layer.output_size()]);
        }

        {
            let input_buf = self.input_buf.as_mut().unwrap();
            let mut dest = input_buf.as_mut_ptr();
            let mut feature_map = (*self.feature_map).borrow_mut();
            unsafe {
                for &feature_id in &self.feature_ids[..] {
                    feature_map.get_feature_vector((feature_id - 1) as usize, dest);
                    let offset = feature_map.get_feature_vector_dim(feature_id as usize);
                    dest = dest.offset(offset as isize);
                }
            }
        }

        self.compute_internal();

        let output_buf = self.output_buf.as_ref().unwrap();
        let score = *output_buf.get(0).expect("No score");
        let score = Score {
            positive: score > self.thresh,
            score,
        };

        if let Some(output) = output {
            unsafe {
                ptr::copy_nonoverlapping(
                    output_buf.as_ptr(),
                    output.as_mut_ptr(),
                    output_buf.len(),
                );
            }
        }

        score
    }

    pub fn compute(&mut self, image: &ImageData) {
        (*self.feature_map)
            .borrow_mut()
            .compute(image.data(), image.width(), image.height());
    }

    pub fn set_roi(&mut self, roi: Rectangle) {
        (*self.feature_map).borrow_mut().set_roi(roi);
    }
}
