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

pub struct FaceInfo {
    bbox: Rectangle,
    roll: f64,
    pitch: f64,
    yaw: f64,
    score: f64,
}