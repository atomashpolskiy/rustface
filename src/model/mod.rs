mod classifier;
mod lab_boosted_classifier;

use std::fs::File;
use std::io;
use std::io::{Cursor, Read};

use byteorder::{ReadBytesExt, BigEndian};

pub trait ModelReader {
    fn read(&self, rdr: &mut Cursor<Vec<u8>>) -> Result<(), io::Error>;
}

pub struct Model {

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
                let classifier = classifier::create_classifer(classifier_kind_id);


            }
        }
    }

    let model: Model = Model {};
    Ok(model)
}

fn read_i32(rdr: &mut Cursor<Vec<u8>>) -> Result<i32, io::Error> {
    rdr.read_i32::<BigEndian>()
}