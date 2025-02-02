#[derive(Debug, Clone)]
pub struct PointLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub range: f32,
    pub intensity: f32,
}

pub enum Light {
    Point(PointLight),
}
