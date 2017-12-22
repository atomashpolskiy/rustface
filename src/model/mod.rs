use std::fs::File;
use std::io;
use std::io::{Cursor, Read};
use std::rc::Rc;
use std::cell::RefCell;

use byteorder::{ReadBytesExt, LittleEndian};
use classifier::{Classifier, ClassifierKind, LabBoostedClassifier, SurfMlpClassifier};
use feat::{LabBoostedFeatureMap, SurfMlpFeatureMap};

pub struct Model {
    classifiers: Vec<Box<Classifier>>,
    wnd_src_id: Vec<Vec<i32>>,
    hierarchy_sizes: Vec<i32>,
}

impl Model {
    fn new() -> Self {
        Model {
            classifiers: vec![],
            wnd_src_id: vec![],
            hierarchy_sizes: vec![],
        }
    }

    pub fn get_classifiers(&mut self) -> &mut Vec<Box<Classifier>> {
        &mut self.classifiers
    }

    pub fn get_hierarchy_count(&self) -> usize {
        self.hierarchy_sizes.len()
    }

    pub fn get_hierarchy_size(&self, hierarchy_index: usize) -> i32 {
        self.hierarchy_sizes[hierarchy_index]
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
    lab_boosted_feature_map: Rc<RefCell<LabBoostedFeatureMap>>,
    surf_mlp_feature_map: Rc<RefCell<SurfMlpFeatureMap>>,
}

impl ModelReader {
    fn new(buf: Vec<u8>) -> Self {
        ModelReader {
            reader: Cursor::new(buf),
            lab_boosted_feature_map: Rc::new(RefCell::new(LabBoostedFeatureMap::new())),
            surf_mlp_feature_map: Rc::new(RefCell::new(SurfMlpFeatureMap::new())),
        }
    }

    pub fn read(mut self) -> Result<Model, io::Error> {
        let mut model: Model = Model::new();

        let num_hierarchy = self.read_i32()? as usize;
        model.hierarchy_sizes.resize(num_hierarchy, 0);
        let mut num_stages = Vec::with_capacity(num_hierarchy * 4);

        for _ in 0..num_hierarchy {
            let hierarchy_size = self.read_i32()?;
            model.hierarchy_sizes.push(hierarchy_size);

            for _ in 0..hierarchy_size {
                let num_stage = self.read_i32()?;
                num_stages.push(num_stage);

                for _ in 0..num_stage {
                    let classifier_kind_id = self.read_i32()?;
                    let classifier_kind = ClassifierKind::from(classifier_kind_id);

                    match classifier_kind {
                        Some(classifier_kind) => {
                            model.classifiers.push(self.create_classifier(classifier_kind)?);
                        },
                        None => panic!("Unexpected classifier kind id: {}", classifier_kind_id)
                    };
                }

                let num_wnd_src = self.read_i32()?;
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
                let mut classifier = LabBoostedClassifier::new(Rc::clone(&self.lab_boosted_feature_map));
                self.read_lab_boosted_model(&mut classifier)?;
                Ok(Box::new(classifier))
            },
            ClassifierKind::SurfMlp => {
                let mut classifier = SurfMlpClassifier::new(Rc::clone(&self.surf_mlp_feature_map));
                self.read_surf_mlp_model(&mut classifier)?;
                Ok(Box::new(classifier))
            },
        }
    }

    fn read_lab_boosted_model(&mut self, classifier: &mut LabBoostedClassifier) -> Result<(), io::Error> {
        let num_base_classifier = self.read_i32()?;
        let num_bin = self.read_i32()?;

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
        let num_layer = self.read_i32()?;
        let num_feat = self.read_i32()?;

        for _ in 0..num_feat {
            classifier.add_feature_id(self.read_i32()?);
        }

        classifier.set_threshold(self.read_f32()?);

        let mut input_dim = self.read_i32()?;
        for i in 1..num_layer {
            let output_dim = self.read_i32()?;

            let weights_count = input_dim * output_dim;
            let mut weights: Vec<f32> = Vec::with_capacity(weights_count as usize);
            for _ in 0..weights_count {
                weights.push(self.read_f32()?);
            }

            let mut biases: Vec<f32> = Vec::with_capacity(output_dim as usize);
            for _ in 0..output_dim {
                biases.push(self.read_f32()?);
            }

            if i == num_layer - 1 {
                classifier.add_output_layer(input_dim as usize, output_dim as usize, weights, biases);
            } else {
                classifier.add_layer(input_dim as usize, output_dim as usize, weights, biases);
            }

            input_dim = output_dim;
        }

        Ok(())
    }

    fn read_i32(&mut self) -> Result<i32, io::Error> {
        self.reader.read_i32::<LittleEndian>()
    }

    fn read_f32(&mut self) -> Result<f32, io::Error> {
        self.reader.read_f32::<LittleEndian>()
    }
}



