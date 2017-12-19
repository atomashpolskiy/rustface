mod classifier;
mod lab_boosted_classifier;
mod surf_mlp_classifier;

use std::fs::File;
use std::io;
use std::io::{Cursor, Read};
use std::ops::DerefMut;

use byteorder::{ReadBytesExt, LittleEndian};
use self::classifier::{Classifier, ClassifierKind};
use self::lab_boosted_classifier::LabBoostedClassifier;
use self::surf_mlp_classifier::SurfMlpClassifier;

use std::collections::HashMap;
use feat::FeatureMap;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Model {
    classifiers: Vec<Box<Classifier>>,
    wnd_src_id: Vec<Vec<i32>>,
}

impl Model {
    fn new() -> Self {
        Model {
            classifiers: vec![],
            wnd_src_id: vec![],
        }
    }
}

pub fn load_model(path: &str) -> Result<Model, io::Error> {
    let mut buf = vec![];
    File::open(path).map(|mut file|
        file.read_to_end(&mut buf)
    )??;
    ModelReader::new(buf).read()
}

struct ModelReader {
    reader: Cursor<Vec<u8>>,
    featmaps_by_classifier_kind: HashMap<ClassifierKind, Rc<FeatureMap>>,
}

impl ModelReader {
    fn new(buf: Vec<u8>) -> Self {
        ModelReader {
            reader: Cursor::new(buf),
            featmaps_by_classifier_kind: HashMap::new()
        }
    }

    pub fn read(mut self) -> Result<Model, io::Error> {
        let mut model: Model = Model::new();

        let num_hierarchy = self.read_i32()?;
        let mut hierarchy_sizes = Vec::with_capacity(num_hierarchy as usize);
        let mut num_stages = Vec::with_capacity(hierarchy_sizes.len() * 4);

        println!("num hierarchy: {}", num_hierarchy);
        for _ in 0..num_hierarchy {
            let hierarchy_size = self.read_i32()?;
            hierarchy_sizes.push(hierarchy_size);

            println!("hierarchy size: {}", hierarchy_size);
            for _ in 0..hierarchy_size {
                let num_stage = self.read_i32()?;
                num_stages.push(num_stage);

                println!("num stage: {}", num_stage);
                for _ in 0..num_stage {
                    let classifier_kind_id = self.read_i32()?;
                    let classifier_kind = ClassifierKind::from(classifier_kind_id);

                    println!("classifier kind id: {}", classifier_kind_id);
                    match classifier_kind {
                        Some(classifier_kind) => {
                            model.classifiers.push(self.create_classifier(classifier_kind)?);
                        },
                        None => panic!("Unexpected classifier kind id: {}", classifier_kind_id)
                    };
                }

                let num_wnd_src = self.read_i32()?;
                println!("num wnd src: {}", num_wnd_src);
                let mut num_wnd_vec;
                if num_wnd_src > 0 {
                    num_wnd_vec = Vec::with_capacity(num_wnd_src as usize);
                    for _ in 0..num_wnd_src {
                        num_wnd_vec.push(self.read_i32()?);
                    }
                } else {
                    num_wnd_vec = vec![];
                }
                model.wnd_src_id.push(num_wnd_vec);
            }
        }

        Ok(model)
    }

    fn create_classifier(&mut self, classifier_kind: ClassifierKind) -> Result<Box<Classifier>, io::Error> {
        match classifier_kind {
            ClassifierKind::LabBoosted => {
                let mut classifier;
                {
                    let feature_map = self.get_or_create_feature_map(classifier_kind);
                    classifier = LabBoostedClassifier::new(Rc::clone(feature_map));
                }
                self.read_lab_boosted_model(&mut classifier)?;
                Ok(Box::new(classifier))
            },
            ClassifierKind::SurfMlp => {
                let mut classifier;
                {
                    let feature_map = self.get_or_create_feature_map(classifier_kind);
                    classifier = SurfMlpClassifier::new(Rc::clone(feature_map));
                }
                self.read_surf_mlp_model(&mut classifier)?;
                Ok(Box::new(classifier))
            }
            _ => panic!("Unsupported classifier kind: {:?}", classifier_kind)
        }
    }

    fn get_or_create_feature_map<'a>(&'a mut self, classifier_kind: ClassifierKind) -> &'a mut Rc<FeatureMap> {
        self.featmaps_by_classifier_kind.entry(classifier_kind)
            .or_insert_with(|| Rc::new(FeatureMap::new()))
    }

    fn read_lab_boosted_model(&mut self, classifier: &mut LabBoostedClassifier) -> Result<(), io::Error> {
        let num_base_classifier = self.read_i32()?;
        let num_bin = self.read_i32()?;

        println!("num base classifier: {}", num_base_classifier);
        println!("num base classifier as usize: {}", num_base_classifier as usize);
        println!("num bin: {}", num_bin);
        for _ in 0..num_base_classifier {
            let x = self.read_i32()?;
            let y = self.read_i32()?;
            classifier.add_feature(x, y);
        }

        let mut thresh: Vec<f32> = Vec::with_capacity(num_base_classifier as usize);
        for _ in 0..num_base_classifier {
            thresh.push(self.read_f32()?);
        }

        for i in 0..num_base_classifier {
            let mut weights: Vec<f32> = Vec::with_capacity(num_bin as usize + 1);
            for _ in 0..weights.capacity() {
                weights.push(self.read_f32()?);
            }
            classifier.add_base_classifier(weights, thresh[i as usize]);
        }

        Ok(())
    }

    fn read_surf_mlp_model(&mut self, classifier: &mut SurfMlpClassifier) -> Result<(), io::Error> {


        Ok(())
    }

    fn read_i32(&mut self) -> Result<i32, io::Error> {
        self.reader.read_i32::<LittleEndian>()
    }

    fn read_f32(&mut self) -> Result<f32, io::Error> {
        self.reader.read_f32::<LittleEndian>()
    }
}



