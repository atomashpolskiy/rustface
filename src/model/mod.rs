mod classifier;
mod lab_boosted_classifier;

use std::fs::File;
use std::io;
use std::io::{Cursor, Read};

use byteorder::{ReadBytesExt, BigEndian};
use self::classifier::{Classifier, ClassifierKind};
use self::lab_boosted_classifier::LabBoostedClassifier;
use std::collections::HashMap;
use feat::FeatureMap;
use std::rc::Rc;

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

struct ModelLoader {
    featmaps_by_classifier_kind: HashMap<ClassifierKind, Rc<FeatureMap>>,
}

impl ModelLoader {
    fn new() -> Self {
        ModelLoader {
            featmaps_by_classifier_kind: HashMap::new()
        }
    }

    fn load_model(&mut self, path: &str) -> Result<Model, io::Error> {
        let mut buf = vec![];
        File::open(path).map(|mut file|
            file.read_to_end(&mut buf)
        )?;

        let mut model: Model = Model::new();
        let mut rdr = Cursor::new(buf);

        let num_hierarchy = Self::read_i32(&mut rdr)?;
        let mut hierarchy_sizes = Vec::with_capacity(num_hierarchy as usize);
        let mut num_stages = Vec::with_capacity(hierarchy_sizes.len() * 4);

        for _ in 0..num_hierarchy {
            let hierarchy_size = Self::read_i32(&mut rdr)?;
            hierarchy_sizes.push(hierarchy_size);

            for _ in 0..hierarchy_size {
                let num_stage = Self::read_i32(&mut rdr)?;
                num_stages.push(num_stage);

                for _ in 0..num_stage {
                    let classifier_kind_id = Self::read_i32(&mut rdr)?;
                    let classifier_kind = ClassifierKind::from(classifier_kind_id);

                    match classifier_kind {
                        Some(classifier_kind) => {
                            model.classifiers.push(self.create_classifer(classifier_kind, &mut rdr)?);
                        },
                        None => panic!("Unexpected classifier kind id: {}", classifier_kind_id)
                    };
                }

                let num_wnd_src = Self::read_i32(&mut rdr)?;
                let mut num_wnd_vec = Vec::with_capacity(num_wnd_src as usize);
                if num_wnd_src > 0 {
                    for _ in 0..num_wnd_src {
                        num_wnd_vec.push(Self::read_i32(&mut rdr)?);
                    }
                }
                model.wnd_src_id.push(num_wnd_vec);
            }
        }

        Ok(model)
    }

    fn create_classifer(&mut self, classifier_kind: ClassifierKind, mut rdr: &mut Cursor<Vec<u8>>) -> Result<Box<Classifier>, io::Error> {
        match classifier_kind {
            ClassifierKind::LabBoostedClassifier => {
                let feature_map = self.featmaps_by_classifier_kind.entry(classifier_kind)
                    .or_insert_with(|| Rc::new(FeatureMap::new()));
                let mut classifier = LabBoostedClassifier::new(Rc::clone(feature_map));
                Self::read_lab_boosted_model(&mut rdr, &mut classifier)?;
                Ok(Box::new(classifier))
            },
            _ => panic!("Unsupported classifier kind: {:?}", classifier_kind)
        }
    }

    fn read_lab_boosted_model(mut rdr: &mut Cursor<Vec<u8>>, classifier: &mut LabBoostedClassifier) -> Result<(), io::Error> {
        let num_base_classifier = Self::read_i32(&mut rdr)?;
        let num_bin = Self::read_i32(&mut rdr)?;

        for _ in 0..num_base_classifier {
            let x = Self::read_i32(&mut rdr)?;
            let y = Self::read_i32(&mut rdr)?;
            classifier.add_feature(x, y);
        }

        let mut thresh: Vec<f32> = Vec::with_capacity(num_base_classifier as usize);
        for _ in 0..num_base_classifier {
            thresh.push(Self::read_f32(&mut rdr)?);
        }

        for i in 0..num_base_classifier {
            let mut weights: Vec<f32> = Vec::with_capacity(num_bin as usize + 1);
            for _ in 0..weights.capacity() {
                weights.push(Self::read_f32(&mut rdr)?);
            }
            classifier.add_base_classifier(weights, thresh[i as usize]);
        }

        Ok(())
    }

    fn read_i32(rdr: &mut Cursor<Vec<u8>>) -> Result<i32, io::Error> {
        rdr.read_i32::<BigEndian>()
    }

    fn read_f32(rdr: &mut Cursor<Vec<u8>>) -> Result<f32, io::Error> {
        rdr.read_f32::<BigEndian>()
    }
}



