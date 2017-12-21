use math;
use common::Rectangle;
use feat::FeatureMap;

pub struct LabBoostedFeatureMap {
    roi: Option<Rectangle>,
    width: u32,
    height: u32,
    length: usize,
    feat_map: Vec<u8>,
    rect_sum: Vec<i32>,
    int_img: Vec<i32>,
    square_int_img: Vec<i32>,
    rect_width: u32,
    rect_height: u32,
    num_rect: u32,
}

impl FeatureMap for LabBoostedFeatureMap {
    fn compute(&mut self, input: *const u8, width: u32, height: u32) {
        if width == 0 || height == 0 {
            panic!(format!("Illegal arguments: width ({}), height ({})", width, height));
        }

        self.reshape(width, height);
        self.compute_integral_images(input);
        self.compute_rect_sum();
        self.compute_feature_map();
    }
}

impl LabBoostedFeatureMap {
    pub fn new() -> Self {
        LabBoostedFeatureMap {
            roi: None,
            width: 0,
            height: 0,
            length: 0,
            feat_map: vec![],
            rect_sum: vec![],
            int_img: vec![],
            square_int_img: vec![],
            rect_width: 0,
            rect_height: 0,
            num_rect: 0,
        }
    }

    pub fn get_feature_val(&self, offset_x: u32, offset_y: u32) -> u8 {
        let roi = self.roi.as_ref().unwrap();
        let i = (roi.y() + offset_y) * self.width + roi.x() + offset_x;
        self.feat_map.get(i as usize)
            .expect(&format!("requested element #{}, but length is {}", i, self.feat_map.len())[..])
            .clone()
    }

    pub fn get_std_dev(&self) -> f64 {
        let roi = self.roi.as_ref().unwrap();

        let mean: f64;
        let m2: f64;
        let area: f64 = (roi.width() * roi.height()) as f64;

        match (roi.x(), roi.y()) {
            (0, 0) => {
                let bottom_right = (roi.height() - 1) * self.width + roi.width() - 1;
                mean = self.int_img[bottom_right as usize] as f64 / area;
                m2 = self.square_int_img[bottom_right as usize] as f64 / area;
            }
            (0, _) => {
                let top_right = (roi.y() - 1) * self.width + roi.width() - 1;
                let bottom_right = top_right + roi.height() * self.width;
                mean = (self.int_img[bottom_right as usize] - self.int_img[top_right as usize]) as f64 / area;
                m2 = (self.square_int_img[bottom_right as usize] - self.square_int_img[top_right as usize]) as f64 / area;
            }
            (_, 0) => {
                let bottom_left = (roi.height() - 1) * self.width + roi.x() - 1;
                let bottom_right = bottom_left + roi.width();
                mean = (self.int_img[bottom_right as usize] - self.int_img[bottom_left as usize]) as f64 / area;
                m2 = (self.square_int_img[bottom_right as usize] - self.square_int_img[bottom_left as usize]) as f64 / area;
            }
            (_, _) => {
                let top_left = (roi.y() - 1) * self.width + roi.x() - 1;
                let top_right = top_left + roi.width();
                let bottom_left = top_left + roi.height() * self.width;
                let bottom_right = bottom_left + roi.width();
                mean = (self.int_img[bottom_right as usize] - self.int_img[bottom_left as usize] +
                    self.int_img[top_left as usize] - self.int_img[top_right as usize]) as f64 / area;
                m2 = (self.square_int_img[bottom_right as usize] - self.square_int_img[bottom_left as usize] +
                    self.square_int_img[top_left as usize] - self.square_int_img[top_right as usize]) as f64 / area;
            }
        }

        (m2 - mean * mean).sqrt()
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

            LabBoostedFeatureMap::compute_integral(self.int_img.as_mut_ptr(), self.width, self.height);
            LabBoostedFeatureMap::compute_integral(self.square_int_img.as_mut_ptr(), self.width, self.height);
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

    fn compute_rect_sum(&mut self) {
        let width = (self.width - self.rect_width) as usize;
        let height = self.height - self.rect_height;

        let int_img_ptr = self.int_img.as_ptr();
        let rect_sum_ptr = self.rect_sum.as_mut_ptr();

        unsafe {
            *rect_sum_ptr = *(int_img_ptr.offset(((self.rect_height - 1) * self.width + self.rect_width - 1) as isize));
            math::vector_sub(
                int_img_ptr.offset(((self.rect_height - 1) * self.width + self.rect_width) as isize),
                int_img_ptr.offset(((self.rect_height - 1) * self.width) as isize),
                rect_sum_ptr.offset(1),
                width);

            for i in 1..height {
                let top_left = int_img_ptr.offset(((i - 1) * self.width) as isize);
                let top_right = top_left.offset((self.rect_width - 1) as isize);
                let bottom_left = top_left.offset((self.rect_height * self.width) as isize);
                let bottom_right = bottom_left.offset((self.rect_width - 1) as isize);

                let mut dest = rect_sum_ptr.offset((i * self.width) as isize);
                *dest = *bottom_right - *top_right;
                dest = dest.offset(1);

                math::vector_sub(bottom_right.offset(1), top_right.offset(1), dest, width);
                math::vector_sub(dest, bottom_left, dest, width);
                math::vector_add(dest, top_left, dest, width);
            }
        }
    }

    fn compute_feature_map(&mut self) {
        let width = self.width - self.rect_width * self.num_rect;
        let height = self.height - self.rect_height * self.num_rect;
        let offset = self.width * self.rect_height;

        let feat_map_ptr = self.feat_map.as_mut_ptr();

        unsafe {
            for r in 0..height {
                for c in 0..width {
                    let dest = feat_map_ptr.offset((r * self.width + c) as isize);
                    *dest = 0;

                    let white_rect_sum = self.rect_sum[((r + self.rect_height) * self.width + c + self.rect_width) as usize];

                    let mut black_rect_idx = r * self.width + c;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x80
                    };

                    black_rect_idx += self.rect_width;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x40
                    };
                    black_rect_idx += self.rect_width;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x20
                    };

                    black_rect_idx += offset;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x08
                    };
                    black_rect_idx += offset;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x01
                    };

                    black_rect_idx -= self.rect_width;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x02
                    };
                    black_rect_idx -= self.rect_width;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x04
                    };

                    black_rect_idx -= offset;
                    if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                        *dest |= 0x10
                    };
                }
            }
        }
    }
}