pub struct SurfMlpFeatureMap {
    roi: Option<Rectangle>,
    width: u32,
    height: u32,
    length: usize,
}

impl SurfMlpFeatureMap {
    pub fn new() -> Self {
        SurfMlpFeatureMap {
            roi: None,
            width: 0,
            height: 0,
            length: 0,
        }
    }
}