use crate::{
    cubemap::{
        self,
        CubeFace,
    },
    gltf::GltfAdapter,
    image::Image,
    model::{
        Model,
        ModelDescriptor,
        ModelTrait,
    },
};
use anyhow::{
    bail,
    ensure,
    Result,
};
use chroma_base::path::get_project_root;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SceneDescriptor {
    pub scene: String,
    pub models: Vec<ModelDescriptor>,
    pub skybox: Option<std::path::PathBuf>,
}

impl SceneDescriptor {
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        ensure!(path.exists(), "Specified path doesn't exist: {:?}", path);

        let file = std::fs::File::open(path)?;
        let mut deserializer = serde_json::Deserializer::from_reader(file);
        let scene_descriptor = SceneDescriptor::deserialize(&mut deserializer)?;
        Ok(scene_descriptor)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Scene {
    pub name: String,
    pub models: Vec<Model>,
    pub skybox: Option<HashMap<CubeFace, Image>>,
}

impl Scene {
    pub fn from_scene_descriptor(desc: SceneDescriptor) -> Result<Self> {
        let mut models = Vec::new();
        for model_desc in desc.models {
            let model = {
                let gltf = GltfAdapter::from_model_descriptor(&model_desc).unwrap();
                Model::Gltf(gltf)
            };
            models.push(model);
        }

        let skybox = if let Some(path) = desc.skybox {
            let image = Image::from_exr(&path);
            Some(cubemap::convert_equirectangular_to_cubemap_image(&image))
        } else {
            None
        };

        Ok(Self {
            name: desc.scene,
            models,
            skybox,
        })
    }

    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let scene_descriptor = SceneDescriptor::from_path(path)?;
        Self::from_scene_descriptor(scene_descriptor)
    }

    pub fn from_scene_name(scene_name: &str) -> Result<Self> {
        let config_root = get_project_root()
            .unwrap()
            .join("crates/chroma-scene/config");
        for entry in std::fs::read_dir(config_root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let scene_desc = SceneDescriptor::from_path(&path)?;
                if scene_desc.scene == scene_name {
                    return Self::from_path(&path);
                }
            }
        }
        bail!("Scene not found: {}", scene_name);
    }
}
