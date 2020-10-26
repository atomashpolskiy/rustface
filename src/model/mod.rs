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

use crate::classifier::Score;
use std::fs::File;
use std::io::BufReader;
use std::io;

use crate::classifier::{Classifier, ClassifierKind, LabBoostedClassifier, SurfMlpClassifier};
use crate::feat::{FeatureMap, LabBoostedFeatureMap, SurfMlpFeatureMap};
use byteorder::{LittleEndian, ReadBytesExt};

pub struct Model {
    classifiers: Vec<Classifier>,
    wnd_src_id: Vec<Vec<i32>>,
    hierarchy_sizes: Vec<i32>,
    num_stages: Vec<i32>,
    lab_boosted_feature_map: LabBoostedFeatureMap,
    surf_mlp_feature_map: SurfMlpFeatureMap,
}

impl Model {
    pub fn get_classifiers(&mut self) -> &mut Vec<Classifier> {
        &mut self.classifiers
    }

    pub fn feature_map_for_classifier(&mut self, index: usize) -> &mut dyn FeatureMap {
        match self.classifiers[index] {
            Classifier::LabBoosted(_) => &mut self.lab_boosted_feature_map,
            Classifier::SurfMlp(_) => &mut self.surf_mlp_feature_map,
        }
    }

    pub fn classify_with_classifier(&mut self, index: usize, output: Option<&mut Vec<f32>>) -> Score {
        match self.classifiers[index] {
            Classifier::SurfMlp(ref mut c) => c.classify(output, &mut self.surf_mlp_feature_map),
            Classifier::LabBoosted(ref mut c) => c.classify(&mut self.lab_boosted_feature_map),
        }
    }

    pub fn get_wnd_src(&self, id: usize) -> &Vec<i32> {
        &self.wnd_src_id[id]
    }

    pub fn get_hierarchy_count(&self) -> usize {
        self.hierarchy_sizes.len()
    }

    pub fn get_num_stage(&self, id: usize) -> i32 {
        self.num_stages[id]
    }

    pub fn get_hierarchy_size(&self, hierarchy_index: usize) -> i32 {
        self.hierarchy_sizes[hierarchy_index]
    }
}

/// Load model from a file.
pub fn load_model(path: &str) -> Result<Model, io::Error> {
    read_model(BufReader::new(File::open(path)?))
}

/// Load model from any stream or buffer
pub fn read_model<R: io::Read>(buf: R) -> Result<Model, io::Error> {
    ModelReader::new(buf).read()
}

struct ModelReader<R: io::Read> {
    reader: R,
}

impl<R: io::Read> ModelReader<R> {
    fn new(reader: R) -> Self {
        ModelReader {
            reader,
        }
    }

    pub fn read(mut self) -> Result<Model, io::Error> {
        let num_hierarchy = self.read_i32()? as usize;
        let mut classifiers = Vec::new();
        let mut hierarchy_sizes = Vec::with_capacity(num_hierarchy);
        let mut num_stages = Vec::new();
        let mut wnd_src_id = Vec::new();

        for _ in 0..num_hierarchy {
            let hierarchy_size = self.read_i32()?;
            hierarchy_sizes.push(hierarchy_size);

            for _ in 0..hierarchy_size {
                let num_stage = self.read_i32()?;
                num_stages.push(num_stage);

                for _ in 0..num_stage {
                    let classifier_kind_id = self.read_i32()?;
                    let classifier_kind = ClassifierKind::from(classifier_kind_id);

                    match classifier_kind {
                        Some(ref classifier_kind) => {
                            classifiers.push(self.create_classifier(classifier_kind)?);
                        }
                        None => panic!("Unexpected classifier kind id: {}", classifier_kind_id),
                    };
                }

                let num_wnd_src = self.read_i32()?;
                let mut num_wnd_vec;
                if num_wnd_src > 0 {
                    num_wnd_vec = Vec::with_capacity(num_wnd_src as usize);
                    for _ in 0..num_wnd_src {
                        num_wnd_vec.push(self.read_i32()?);
                    }
                } else {
                    num_wnd_vec = vec![];
                }
                wnd_src_id.push(num_wnd_vec);
            }
        }

        Ok(Model {
            lab_boosted_feature_map: LabBoostedFeatureMap::new(),
            surf_mlp_feature_map: SurfMlpFeatureMap::new(),
            classifiers,
            wnd_src_id,
            hierarchy_sizes,
            num_stages,
        })
    }

    fn create_classifier(
        &mut self,
        classifier_kind: &ClassifierKind,
    ) -> Result<Classifier, io::Error> {
        match *classifier_kind {
            ClassifierKind::LabBoosted => {
                let mut classifier = LabBoostedClassifier::new();
                self.read_lab_boosted_model(&mut classifier)?;
                Ok(Classifier::LabBoosted(classifier))
            }
            ClassifierKind::SurfMlp => {
                let mut classifier = SurfMlpClassifier::new();
                self.read_surf_mlp_model(&mut classifier)?;
                Ok(Classifier::SurfMlp(classifier))
            }
        }
    }

    fn read_lab_boosted_model(
        &mut self,
        classifier: &mut LabBoostedClassifier,
    ) -> Result<(), io::Error> {
        let num_base_classifier = self.read_i32()?;
        let num_bin = self.read_i32()?;

        for _ in 0..num_base_classifier {
            let x = self.read_i32()?;
            let y = self.read_i32()?;
            classifier.add_feature(x, y);
        }

        let mut thresh: Vec<f32> = Vec::with_capacity(num_base_classifier as usize);
        for _ in 0..num_base_classifier {
            thresh.push(self.read_f32()?);
        }

        for i in 0..num_base_classifier {
            let mut weights: Vec<f32> = Vec::with_capacity(num_bin as usize + 1);
            for _ in 0..weights.capacity() {
                weights.push(self.read_f32()?);
            }
            classifier.add_base_classifier(weights, thresh[i as usize]);
        }

        Ok(())
    }

    fn read_surf_mlp_model(&mut self, classifier: &mut SurfMlpClassifier) -> Result<(), io::Error> {
        let num_layer = self.read_i32()?;
        let num_feat = self.read_i32()?;

        for _ in 0..num_feat {
            classifier.add_feature_id(self.read_i32()?);
        }

        classifier.set_threshold(self.read_f32()?);

        let mut input_dim = self.read_i32()?;
        for i in 1..num_layer {
            let output_dim = self.read_i32()?;

            let weights_count = input_dim * output_dim;
            let mut weights: Vec<f32> = Vec::with_capacity(weights_count as usize);
            for _ in 0..weights_count {
                weights.push(self.read_f32()?);
            }

            let mut biases: Vec<f32> = Vec::with_capacity(output_dim as usize);
            for _ in 0..output_dim {
                biases.push(self.read_f32()?);
            }

            if i == num_layer - 1 {
                classifier.add_output_layer(
                    input_dim as usize,
                    output_dim as usize,
                    weights,
                    biases,
                );
            } else {
                classifier.add_layer(input_dim as usize, output_dim as usize, weights, biases);
            }

            input_dim = output_dim;
        }

        Ok(())
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32, io::Error> {
        self.reader.read_i32::<LittleEndian>()
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, io::Error> {
        self.reader.read_f32::<LittleEndian>()
    }
}
