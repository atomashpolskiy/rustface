pub struct LabBoostedClassifier {

}

impl LabBoostedClassifier {
    pub fn new() -> Self {
        LabBoostedClassifier {}
    }
}

impl Classifier for LabBoostedClassifier {
    fn classify(&self, features: FeatureMap) -> Option<Score> {
        unimplemented!()
    }
}