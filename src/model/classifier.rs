use feat::FeatureMap;
use super::lab_boosted_classifier::LabBoostedClassifier;

pub fn create_classifer(classifier_kind_id: i32) -> Box<Classifier> {
    let classifier_kind = ClassifierKind::from(classifier_kind_id);
    match classifier_kind {
        Some(ClassifierKind::LabBoostedClassifier) => return Box::new(LabBoostedClassifier::new()),
        Some(_) => panic!("Unsupported classifier kind: {:?}", classifier_kind),
        None => panic!("Unexpected classifier kind id: {}", classifier_kind_id)
    }
}

#[derive(Debug)]
enum ClassifierKind {
    LabBoostedClassifier,
    SurfMlp,
}

impl ClassifierKind {
    pub fn from(id: i32) -> Option<Self> {
        match id {
            1 => Some(ClassifierKind::LabBoostedClassifier),
            2 => Some(ClassifierKind::SurfMlp),
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