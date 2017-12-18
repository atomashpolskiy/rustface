mod math;

pub struct FeatureMap {
    roi: Option<Rectangle>,
    width: u32,
    height: u32,
    length: usize,
    feat_map: Vec<u8>,
    rect_sum: Vec<i32>,
    int_img: Vec<i32>,
    square_int_img: Vec<i32>,
}

impl FeatureMap {
    pub fn new() -> Self {
        FeatureMap {
            roi: None,
            width: 0,
            height: 0,
            length: 0,
            feat_map: vec![],
            rect_sum: vec![],
            int_img: vec![],
            square_int_img: vec![],
        }
    }

    pub fn compute(&mut self, input: *const u8, width: u32, height: u32) {
        if width == 0 || height == 0 {
            panic!(format!("Illegal arguments: width ({}), height ({})", width, height));
        }

        self.reshape(width, height);
        self.compute_integral_images(input);
    }

    fn reshape(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.length = (width * height) as usize;

        self.feat_map.resize(self.length, 0);
        self.rect_sum.resize(self.length, 0);
        self.int_img.resize(self.length, 0);
        self.square_int_img.resize(self.length, 0);
    }

    fn compute_integral_images(&mut self, input: *const u8) {
        unsafe {
            math::copy_u8_to_i32(input, self.int_img.as_mut_ptr(), self.length);
            math::square(self.int_img.as_ptr(), self.square_int_img.as_mut_ptr(), self.length);

            compute_integral(self.int_img.as_mut_ptr(), self.width, self.height);
            compute_integral(self.square_int_img.as_mut_ptr(), self.width, self.height);
        }
    }
}

unsafe fn compute_integral(data: *mut i32, width: u32, height: u32) {
    let mut src = data;
    let mut dest = data;
    let mut dest_previous_row = dest;

    src = src.offset(1);
    for _ in 1..width {
        *dest.offset(1) = *dest + *src;

        src = src.offset(1);
        dest = dest.offset(1);
    }

    dest = dest.offset(1);
    for _ in 1..height {
        let mut s = 0;
        for _ in 0..width {
            s += *src;
            *dest = *dest_previous_row + s;

            src = src.offset(1);
            dest = dest.offset(1);
            dest_previous_row = dest_previous_row.offset(1);
        }
    }
}

struct Rectangle {
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
}

pub struct Score {
    score: f32,
    output: f32,
}

pub trait Classify {
    fn classify(features: FeatureMap) -> Option<Score>;
}