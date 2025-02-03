use crate::{
    gltf::GltfAdapter,
    image::Image,
    light::Light,
    material::Volume,
    transform::Transform,
};
use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelDescriptor {
    pub path: std::path::PathBuf,
    pub transform: Transform,
}

#[derive(Debug, Clone)]
pub enum Model {
    Gltf(GltfAdapter),
}

pub trait ModelTrait: Sized {
    fn from_model_descriptor(model_descriptor: &ModelDescriptor) -> Result<Self>;
    fn read_indices(&self) -> Vec<Vec<u32>>;
    fn read_normals(&self) -> Vec<Vec<[f32; 3]>>;
    fn read_positions(&self) -> Vec<Vec<[f32; 3]>>;
    fn read_uvs(&self) -> Vec<Option<Vec<[f32; 2]>>>;
    fn read_tangents(&self) -> Vec<Option<Vec<[f32; 4]>>>;
    fn read_colors(&self) -> Vec<Option<Vec<[f32; 4]>>>;
    fn read_base_colors(&self) -> (Vec<Option<Image>>, Vec<[f32; 4]>);
    fn read_normal_images(&self) -> Vec<Option<Image>>;
    fn read_occlusion_images(&self) -> Vec<Image>;
    fn read_emissive_images(&self) -> Vec<Option<Image>>;
    fn read_metallic_roughnesses(&self) -> (Vec<Option<Image>>, Vec<f32>, Vec<f32>);
    fn read_transmission(&self) -> (Vec<Option<Image>>, Vec<f32>);
    fn path(&self) -> &std::path::PathBuf;
    fn read_volume(&self) -> Vec<Option<Volume>>;
    fn read_punctual_lights(&self) -> Vec<Option<Light>>;
}
