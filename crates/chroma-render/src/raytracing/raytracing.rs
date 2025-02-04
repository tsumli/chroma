use ash::vk;

struct RaytracingResources<'a> {
    pub bottom_level_acceleration_structure: Vec<vk::AccelerationStructureKHR>,
    pub top_level_acceleration_structure: Vec<vk::AccelerationStructureKHR>,
    pub geometry_nodes: Vec<vk::AccelerationStructureGeometryKHR<'a>>,
}
