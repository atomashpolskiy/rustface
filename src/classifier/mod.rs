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

mod lab_boosted_classifier;
mod surf_mlp_classifier;

pub use self::lab_boosted_classifier::LabBoostedClassifier;
pub use self::surf_mlp_classifier::SurfMlpBuffers;
pub use self::surf_mlp_classifier::SurfMlpClassifier;

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum ClassifierKind {
    LabBoosted,
    SurfMlp,
}

impl ClassifierKind {
    #[inline]
    pub fn from(id: i32) -> Option<Self> {
        match id {
            0 => Some(ClassifierKind::LabBoosted),
            1 => Some(ClassifierKind::SurfMlp),
            _ => None,
        }
    }
}

pub struct Score {
    positive: bool,
    score: f32,
}

impl Score {
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.positive
    }

    #[inline]
    pub fn score(&self) -> f32 {
        self.score
    }
}

#[derive(Clone)]
pub enum Classifier {
    SurfMlp(SurfMlpClassifier),
    LabBoosted(LabBoostedClassifier),
}
