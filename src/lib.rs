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

pub struct Model {

}

#[derive(Debug)]
enum ClassifierKind {
    LabBoostedClassifier,
    SurfMlp,
}

impl ClassifierKind {
    fn from(id: i32) -> Option<Self> {
        use ClassifierKind::*;
        match id {
            1 => Some(LabBoostedClassifier),
            2 => Some(SurfMlp),
            _ => None,
        }
    }
}

fn load_model(path: &str) -> Result<Model, io::Error> {
    let mut buf = vec![];
    File::open(path).map(|mut file|
        file.read_to_end(&mut buf)
    )?;

    let mut rdr = Cursor::new(buf);

    let num_hierarchy = read_i32(&mut rdr)?;
    let mut hierarchy_sizes = Vec::with_capacity(num_hierarchy as usize);
    let mut num_stages = Vec::with_capacity(hierarchy_sizes.len() * 4);

    for i in 0..num_hierarchy {
        let hierarchy_size = read_i32(&mut rdr)?;
        hierarchy_sizes.push(hierarchy_size);

        for j in 0..hierarchy_size {
            let num_stage = read_i32(&mut rdr)?;
            num_stages.push(num_stage);

            for k in 0..num_stage {
                let classifier_kind_id = read_i32(&mut rdr)?;
                let classifier = create_classifer(classifier_kind_id);
            }
        }
    }

    let model: Model = Model {};
    Ok(model)
}

fn read_i32(rdr: &mut Cursor<Vec<u8>>) -> Result<i32, io::Error> {
    rdr.read_i32::<BigEndian>()
}

fn create_classifer(classifier_kind_id: i32) -> Box<Classifier> {
    let classifier_kind = ClassifierKind::from(classifier_kind_id);
    match classifier_kind {
        Some(ClassifierKind::LabBoostedClassifier) => return Box::new(LabBoostedClassifier::new()),
        Some(_) => panic!("Unsupported classifier kind: {:?}", classifier_kind),
        None => panic!("Unexpected classifier kind id: {}", classifier_kind_id)
    }
}



pub struct Score {
    score: f32,
    output: f32,
}

pub trait Classifier {
    fn classify(&self, features: FeatureMap) -> Option<Score>;
}

struct LabBoostedClassifier {

}

impl LabBoostedClassifier {
    fn new() -> Self {
        LabBoostedClassifier {}
    }
}

impl Classifier for LabBoostedClassifier {
    fn classify(&self, features: FeatureMap) -> Option<Score> {
        unimplemented!()
    }
}
