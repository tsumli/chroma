use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Transform {
    pub scale: f32,
    pub translation: [f32; 3],
    pub rotation: [f32; 3],
}
