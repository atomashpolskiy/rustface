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

use super::Score;
use crate::feat::SurfMlpFeatureMap;
use crate::math;
use crate::Rectangle;
use std::mem;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

struct TwoWayBuffer {
    input: Vec<f32>,
    output: Vec<f32>,
}

impl TwoWayBuffer {
    #[inline]
    fn new() -> Self {
        TwoWayBuffer {
            input: Vec::new(),
            output: Vec::new(),
        }
    }

    #[inline]
    fn get_buffers(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>) {
        (&mut self.input, &mut self.output)
    }

    #[inline]
    fn get_input(&mut self) -> &mut Vec<f32> {
        &mut self.input
    }

    #[inline]
    fn swap(&mut self) {
        mem::swap(&mut self.input, &mut self.output);
    }
}

pub struct SurfMlpBuffers {
    input: Vec<f32>,
    output: Vec<f32>,
    layers: TwoWayBuffer,
}

impl SurfMlpBuffers {
    #[inline]
    pub fn new() -> Self {
        SurfMlpBuffers {
            input: Vec::new(),
            output: Vec::new(),
            layers: TwoWayBuffer::new(),
        }
    }
}

#[derive(Clone)]
pub struct SurfMlpClassifier {
    feature_ids: Vec<i32>,
    thresh: f32,
    layers: Vec<Layer>,
}

impl SurfMlpClassifier {
    #[inline]
    pub fn new() -> Self {
        SurfMlpClassifier {
            feature_ids: Vec::new(),
            thresh: 0.0,
            layers: Vec::new(),
        }
    }

    #[inline]
    pub fn add_feature_id(&mut self, feature_id: i32) {
        self.feature_ids.push(feature_id);
    }

    #[inline]
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
            act_func: Self::relu,
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
            act_func: Self::sigmoid,
        })
    }

    #[inline]
    fn relu(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn compute_internal(&self, bufs: &mut SurfMlpBuffers) {
        bufs.layers
            .get_input()
            .resize(self.layers[0].output_size(), 0.0);
        self.layers[0].compute(&bufs.input, bufs.layers.get_input());

        for i in 1..(self.layers.len() - 1) {
            {
                let layer = &self.layers[i];
                let (input_buf, output_buf) = bufs.layers.get_buffers();
                output_buf.resize(layer.output_size(), 0.0);
                layer.compute(input_buf, output_buf);
            }
            bufs.layers.swap();
        }

        let last_layer = &self.layers[self.layers.len() - 1];
        last_layer.compute(bufs.layers.get_input(), &mut bufs.output);
    }
}

#[derive(Clone)]
struct Layer {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    act_func: fn(f32) -> f32,
}

impl Layer {
    fn compute(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "rayon")]
        let it = self.weights.par_chunks(self.input_dim);

        #[cfg(not(feature = "rayon"))]
        let it = self.weights.chunks(self.input_dim);

        it.zip(&self.biases)
            .zip(output)
            .for_each(|((weights, bias), output)| {
                let x = math::vector_inner_product(input, weights) + bias;
                *output = (self.act_func)(x);
            });
    }

    #[inline]
    fn input_size(&self) -> usize {
        self.input_dim
    }

    #[inline]
    fn output_size(&self) -> usize {
        self.output_dim
    }
}

impl SurfMlpClassifier {
    pub fn classify(
        &self,
        output: Option<&mut Vec<f32>>,
        bufs: &mut SurfMlpBuffers,
        feature_map: &mut SurfMlpFeatureMap,
        roi: Rectangle,
    ) -> Score {
        let input_layer = self.layers.get(0).expect("No layers");
        bufs.input.resize(input_layer.input_size(), 0.0);

        let num_layers = self.layers.len();
        let output_layer = self.layers.get(num_layers - 1).expect("No layers");
        bufs.output.resize(output_layer.output_size(), 0.0);

        {
            let mut dest = bufs.input.as_mut_ptr();
            unsafe {
                for &feature_id in &self.feature_ids[..] {
                    feature_map.get_feature_vector((feature_id - 1) as usize, dest, roi);
                    let offset = feature_map.get_feature_vector_dim(feature_id as usize);
                    dest = dest.offset(offset as isize);
                }
            }
        }

        self.compute_internal(bufs);

        let score = *bufs.output.get(0).expect("No score");
        let score = Score {
            positive: score > self.thresh,
            score,
        };

        if let Some(output) = output {
            output.clear();
            output.extend_from_slice(&bufs.output);
        }

        score
    }
}
