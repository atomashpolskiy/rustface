pub mod lab_boosted_featmap;
pub mod surf_mlp_featmap;

pub trait FeatureMap {
    fn compute(&mut self, input: *const u8, width: u32, height: u32);
}