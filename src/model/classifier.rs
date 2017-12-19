fn create_classifer(classifier_kind_id: i32) -> Box<Classifier> {
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
    fn from(id: i32) -> Option<Self> {
        use ClassifierKind::*;
        match id {
            1 => Some(LabBoostedClassifier),
            2 => Some(SurfMlp),
            _ => None,
        }
    }
}

pub trait Classifier {
    fn classify(&self, features: FeatureMap) -> Option<Score>;
}