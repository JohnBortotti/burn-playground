mod model;
mod training;
mod data;
mod inference;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, ndarray::{NdArray, NdArrayDevice}},
    optim::AdamConfig,
    data::dataloader::Dataset
};
fn main() {
    type MyBackend = NdArray<f32, i32>;
    type MyAutoDiffBackend = Autodiff<MyBackend>;

    let device = NdArrayDevice::default();
    let artifact_dir = "./data";

    // crate::training::train::<MyAutoDiffBackend>(
    //     artifact_dir,
    //     TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    //     device.clone(),
    // );

    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test().get(7).unwrap(),
    );
}
