use nalgebra_glm::Vec4;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct MaterialParams {
    pub base_color_factor: Vec4,
    pub metallic_roughness_transmission_factor: Vec4,
}
