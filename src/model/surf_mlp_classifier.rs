use feat::FeatureMap;
use std::rc::Rc;
use super::classifier::{Classifier, Score};

pub struct SurfMlpClassifier {
    feature_map: Rc<FeatureMap>,
}

impl SurfMlpClassifier {
    pub fn new(feature_map: Rc<FeatureMap>) -> Self {
        SurfMlpClassifier {
            feature_map,
        }
    }
}

impl Classifier for SurfMlpClassifier {
    fn classify(&self, features: FeatureMap) -> Option<Score> {
        unimplemented!()
    }
}