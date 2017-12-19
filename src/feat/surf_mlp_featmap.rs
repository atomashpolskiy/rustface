use common::Rectangle;

pub struct SurfMlpFeatureMap {
    roi: Option<Rectangle>,
    width: u32,
    height: u32,
    length: usize,
    buf_valid_reset: bool,
    feature_pool: FeaturePool,
}

impl SurfMlpFeatureMap {
    pub fn new() -> Self {
        let mut map = SurfMlpFeatureMap {
            roi: None,
            width: 0,
            height: 0,
            length: 0,
            buf_valid_reset: false,
            feature_pool: FeaturePool::new(),
        };
        map.init_feature_pool();
        map
    }

    fn init_feature_pool(&mut self) {
        self.feature_pool.add_patch_format(1, 1, 2, 2);
        self.feature_pool.add_patch_format(1, 2, 2, 2);
        self.feature_pool.add_patch_format(2, 1, 2, 2);
        self.feature_pool.add_patch_format(2, 3, 2, 2);
        self.feature_pool.add_patch_format(3, 2, 2, 2);
        self.feature_pool.create();
    }
}

struct FeaturePool {
    sample_width: u32,
    sample_height: u32,
    patch_move_step_x: u32,
    patch_move_step_y: u32,
    patch_size_inc_step: u32,
    patch_min_width: u32,
    patch_min_height: u32,
    features: Vec<Feature>,
    patch_formats: Vec<PatchFormat>,
}

impl FeaturePool {
    fn new() -> Self {
        FeaturePool {
            sample_width: 0,
            sample_height: 0,
            patch_move_step_x: 0,
            patch_move_step_y: 0,
            patch_size_inc_step: 0,
            patch_min_width: 0,
            patch_min_height: 0,
            features: vec![],
            patch_formats: vec![],
        }
    }

    fn add_patch_format(&mut self, width: u32, height: u32, num_cell_per_row: u32, num_cell_per_col: u32) {
        self.patch_formats.push(
            PatchFormat { width, height, num_cell_per_row, num_cell_per_col }
        );
    }

    fn create(&mut self) {
        let mut feature_vecs = vec![];

        if self.sample_height - self.patch_min_height <= self.sample_width - self.patch_min_width {
            for ref format in self.patch_formats.iter() {
                let mut h = self.patch_min_height;
                while h <= self.sample_height {
                    if h % format.num_cell_per_col != 0 || h % format.height != 0 {
                        continue;
                    }
                    let w = h / format.height * format.width;
                    if w % format.num_cell_per_row != 0 || w < self.patch_min_width || w > self.sample_width {
                        continue;
                    }
                    self.collect_features(w, h, format.num_cell_per_row, format.num_cell_per_col, &mut feature_vecs);
                    h += self.patch_size_inc_step;
                }
            }
        } else {
            for ref format in self.patch_formats.iter() {
                let mut w = self.patch_min_width;
                // original condition was <= self.patch_min_width,
                // but it would not make sense to have a loop in such case
                while w <= self.sample_width {
                    if w % format.num_cell_per_row != 0 || w % format.width != 0 {
                        continue;
                    }
                    let h = w / format.width * format.height;
                    if h % format.num_cell_per_col != 0 || h < self.patch_min_height || h > self.sample_height {
                        continue;
                    }
                    self.collect_features(w, h, format.num_cell_per_row, format.num_cell_per_col, &mut feature_vecs);
                    w += self.patch_size_inc_step;
                }
            }
        }

        self.features.append(&mut feature_vecs);
    }

    fn collect_features(&self, width: u32, height: u32, num_cell_per_row: u32, num_cell_per_col: u32, dest: &mut Vec<Feature>) {
        let y_lim = self.sample_height - height;
        let x_lim = self.sample_width - width;

        let mut y = 0;
        while y <= y_lim {
            let mut x = 0;
            while x <= x_lim {
                dest.push(
                    Feature {
                        patch: Rectangle::new(x, y, width, height),
                        num_cell_per_row, num_cell_per_col
                    }
                );
                x += self.patch_move_step_x;
            }
            y += self.patch_move_step_y;
        }
    }
}

struct PatchFormat {
    width: u32,
    height: u32,
    num_cell_per_row: u32,
    num_cell_per_col: u32,
}

struct Feature {
    patch: Rectangle,
    num_cell_per_row: u32,
    num_cell_per_col: u32,
}