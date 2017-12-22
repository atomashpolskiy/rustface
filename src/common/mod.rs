mod image_pyramid;

pub use self::image_pyramid::{ImageData, ImagePyramid, resize_image};

pub struct Rectangle {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

impl Rectangle {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Rectangle { x, y, width, height }
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn set_x(&mut self, x: i32) {
        self.x = x;
    }

    pub fn y(&self) -> i32 {
        self.y
    }

    pub fn set_y(&mut self, y: i32) {
        self.y = y;
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn set_width(&mut self, width: u32) {
        self.width = width;
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn set_height(&mut self, height: u32) {
        self.height = height;
    }
}

pub struct FaceInfo {
    bbox: Rectangle,
    roll: f64,
    pitch: f64,
    yaw: f64,
    score: f64,
}

impl FaceInfo {
    pub fn new() -> Self {
        FaceInfo {
            bbox: Rectangle::new(0, 0, 0, 0),
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            score: 0.0,
        }
    }

    pub fn bbox_mut(&mut self) -> &mut Rectangle {
        &mut self.bbox
    }
}