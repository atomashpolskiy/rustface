mod image_pyramid;

pub struct Rectangle {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Rectangle {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Rectangle { x, y, width, height }
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