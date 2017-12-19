extern crate byteorder;

mod common;
mod math;
mod feat;

use feat::FeatureMap;
use std::fs::File;
use std::io;
use std::io::{Cursor, Read};
use std::error::Error;
use std::string::ToString;

use byteorder::{ReadBytesExt, BigEndian};


pub trait Detector {
//    fn detect(&self, )
}

pub struct Score {
    score: f32,
    output: f32,
}


