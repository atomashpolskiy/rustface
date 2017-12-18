mod math;
mod feat;

use feat::FeatureMap;

pub mod common {
    pub struct Rectangle {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    }

    impl Rectangle {
        pub fn new() -> Self {
            Rectangle {
                x: 0,
                y: 0,
                width: 0,
                height: 0
            }
        }

        pub fn x(&self) -> u32 {
            self.x
        }

        pub fn y(&self) -> u32 {
            self.y
        }

        pub fn width(&self) -> u32 {
            self.width
        }

        pub fn height(&self) -> u32 {
            self.height
        }
    }
}



pub struct Score {
    score: f32,
    output: f32,
}

pub trait Classify {
    fn classify(features: FeatureMap) -> Option<Score>;
}