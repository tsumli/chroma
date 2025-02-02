#[derive(Debug, Clone, Copy)]
pub struct Volume {
    pub thickness_factor: f32,
    pub attenuation_distance: f32,
    pub attenuation_color: [f32; 3],
}
