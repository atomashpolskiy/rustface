use feat::FeatureMap;
use super::lab_boosted_classifier::LabBoostedClassifier;

#[derive(Debug, Hash, PartialEq, Eq)]
pub enum ClassifierKind {
    LabBoostedClassifier,
    SurfMlp,
}

impl ClassifierKind {
    pub fn from(id: i32) -> Option<Self> {
        match id {
            0 => Some(ClassifierKind::LabBoostedClassifier),
            1 => Some(ClassifierKind::SurfMlp),
            _ => None,
        }
    }
}

pub struct Score {
    score: f32,
    output: f32,
}

pub trait Classifier {
    fn classify(&self, features: FeatureMap) -> Option<Score>;
}