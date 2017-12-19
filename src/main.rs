extern crate rustface;

use rustface::model;

fn main() {
    model::load_model("seeta_fd_frontal_v1.0.bin").unwrap();
}
