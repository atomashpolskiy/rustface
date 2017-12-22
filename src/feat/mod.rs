mod lab_boosted_featmap;
mod surf_mlp_featmap;

pub use self::lab_boosted_featmap::LabBoostedFeatureMap;
pub use self::surf_mlp_featmap::SurfMlpFeatureMap;

use common::Rectangle;

pub trait FeatureMap {
    fn set_roi(&mut self, roi: Rectangle);
    fn compute(&mut self, input: *const u8, width: u32, height: u32);
}