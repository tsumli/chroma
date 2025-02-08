use super::{
    raytracing::{
        GeometryNode,
        ShaderBindingTable,
    },
    shadow::ShadowResource,
};
use crate::{
    common::{
        self,
        buffer::{
            allocate_device_local_buffer,
            Buffer,
        },
        camera::{
            CameraParams,
            TransformParams,
        },
        command_buffer::{
            begin_single_time_command,
            end_single_time_command,
        },
        consts::MAX_FRAMES_IN_FLIGHT,
        descriptor_pool,
        device::find_queue_family,
        image_buffer::ImageBuffer,
        index::Index,
        material::MaterialParams,
        pipeline_layout,
        surface,
        swapchain::Swapchain,
        uniform_buffer,
        vertex::Vertex,
    },
    draw::{
        deferred::{
            raytracing::create_identity_transform_matrix,
            shadow::{
                ShadowRenderTarget,
                NUM_SHADOW_DESCRIPTOR_SETS,
            },
        },
        strategy::DrawStrategy,
    },
    shader::shader::{
        create_shader_module,
        read_shader_code,
    },
    texture::{
        texture::Texture,
        texture_sampler::create_texture_sampler,
    },
    utils::{
        math,
        tangent::compute_tangent,
    },
};
use anyhow::{
    ensure,
    Context,
    Result,
};
use ash::vk::{
    self,
    ImageSubresourceRange,
    Packed24_8,
    PhysicalDeviceRayTracingPipelinePropertiesKHR,
    PipelineStageFlags2,
};
use chroma_base::path::get_shader_spv_root;
use chroma_scene::{
    cubemap::CubeFace,
    model::{
        Model,
        ModelTrait as _,
    },
};
use nalgebra_glm::{
    Vec2,
    Vec3,
    Vec4,
};
use std::{
    collections::HashMap,
    ffi::CString,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GraphicsRenderTarget {
    Output,
    Color,
    Normal,
    DepthStencil,
    MetallicRoughness,
    Emissive,
}

const NUM_GRAPHICS_DESCRIPTOR_SETS: usize = 2;
const NUM_COMPUTE_DESCRIPTOR_SETS: usize = 3;
const MIP_LEVEL: u32 = 3;

struct SkyboxResources {
    pub skybox_texture: Texture,
}

#[derive(Clone)]
pub struct AdditionalFunctions {
    pub acceleration_structure: ash::khr::acceleration_structure::Device,
    pub raytracing_pipeline: ash::khr::ray_tracing_pipeline::Device,
    pub mesh_shader: ash::ext::mesh_shader::Device,
    pub get_physical_device_properties2: ash::khr::get_physical_device_properties2::Instance,

    #[allow(dead_code)]
    pub debug_utils: ash::ext::debug_utils::Device,
}

impl AdditionalFunctions {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance, device: &ash::Device) -> Self {
        let acceleration_structure =
            ash::khr::acceleration_structure::Device::new(instance, device);
        let raytracing_pipeline = ash::khr::ray_tracing_pipeline::Device::new(instance, device);
        let mesh_shader = ash::ext::mesh_shader::Device::new(instance, device);
        let get_physical_device_properties2 =
            ash::khr::get_physical_device_properties2::Instance::new(entry, instance);
        let debug_utils = ash::ext::debug_utils::Device::new(instance, device);
        Self {
            acceleration_structure,
            raytracing_pipeline,
            mesh_shader,
            get_physical_device_properties2,
            debug_utils,
        }
    }
}

pub struct Deferred {
    _transform_ubo: uniform_buffer::UniformBuffer<TransformParams>,
    _camera_ubo: uniform_buffer::UniformBuffer<CameraParams>,
    _material_ubo: Vec<uniform_buffer::UniformBuffer<common::material::MaterialParams>>,
    _vertex_buffers: Vec<common::buffer::Buffer>,
    _index_buffers: Vec<common::buffer::Buffer>,
    _base_color_textures: Vec<Texture>,
    _normal_textures: Vec<Texture>,
    _emissive_textures: Vec<Texture>,
    _metallic_roughness_textures: Vec<Texture>,
    graphics_framebuffers: Vec<common::framebuffer::Framebuffer>,
    _command_pool: common::command_pool::CommandPool,
    ash_device: ash::Device,
    _skybox_resources: Option<SkyboxResources>,
    graphics_pipelines: Vec<common::graphics_pipeline::GraphicsPipeline>,
    graphics_pipeline_layouts: Vec<pipeline_layout::PipelineLayout>,
    _graphics_descriptor_pool: descriptor_pool::DescriptorPool,
    graphics_descriptor_sets: Vec<common::descriptor_set::DescriptorSet>,
    graphics_render_pass: common::render_pass::RenderPass,
    _compute_descriptor_pool: descriptor_pool::DescriptorPool,
    compute_descriptor_sets: Vec<common::descriptor_set::DescriptorSet>,
    compute_pipelines: Vec<common::compute_pipeline::ComputePipeline>,
    compute_pipeline_layouts: Vec<pipeline_layout::PipelineLayout>,
    graphics_render_targets: HashMap<GraphicsRenderTarget, ImageBuffer>,
    meshlet_buffers: Vec<common::buffer::Buffer>,
    _meshlet_vertices_buffers: Vec<common::buffer::Buffer>,
    _meshlet_triangle_buffers: Vec<common::buffer::Buffer>,
    additional_functions: AdditionalFunctions,
    _transform_buffers: Vec<Buffer>,
    shadow_resource: ShadowResource,
    shader_group_handle_alignment: u32,
    shader_group_base_alignment: u32,
    shader_group_handle_size: u32,
}

impl Deferred {
    pub fn new(
        transform_ubo: uniform_buffer::UniformBuffer<TransformParams>,
        camera_ubo: uniform_buffer::UniformBuffer<CameraParams>,
        scene: chroma_scene::scene::Scene,
        swapchain: &Swapchain,
        physical_device: vk::PhysicalDevice,
        ash_device: ash::Device,
        surface: &surface::Surface,
        instance: ash::Instance,
        entry: ash::Entry,
    ) -> Result<Self> {
        log::info!("setup additional functions");
        let additional_functions = AdditionalFunctions::new(&entry, &instance, &ash_device);

        log::info!("creating graphics queues");
        let family_indices = find_queue_family(&instance, physical_device, surface)?;
        let graphics_compute_queue = unsafe {
            ash_device.get_device_queue(
                family_indices
                    .graphics_compute_family
                    .context("failed to get graphics family")?,
                0,
            )
        };

        log::info!("creating command pool");
        let command_pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family_indices.graphics_compute_family.unwrap())
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            common::command_pool::CommandPool::new(&command_pool_create_info, ash_device.clone())?
        };

        log::info!("loading models");
        let mut vertex_buffers = Vec::new();
        let mut index_buffers = Vec::new();
        let mut transform_buffers = Vec::new();
        let mut base_color_textures = Vec::new();
        let mut base_color_factors = Vec::new();
        let mut normal_textures = Vec::new();
        let mut transmission_textures = Vec::new();
        let mut transmission_factors = Vec::new();
        let mut metallic_roughness_textures = Vec::new();
        let mut metallic_factors = Vec::new();
        let mut roughness_factors = Vec::new();
        let mut emissive_textures = Vec::new();
        let mut meshlet_buffers = Vec::new();
        let mut meshlet_vertices_buffers = Vec::new();
        let mut meshlet_triangle_buffers = Vec::new();
        let mut geometry_nodes = Vec::new();
        let mut bottom_level_acceleration_structures = Vec::new();
        let mut bottom_level_acceleration_structure_handles = Vec::new();
        let mut bottom_level_acceleration_structure_device_addresses = Vec::new();
        for model in scene.models.iter() {
            match model {
                Model::Gltf(gltf_adapter) => {
                    log::info!("loading gltf model: {:?}", gltf_adapter.path());

                    // vertex buffer
                    log::info!("loading vertex buffer");
                    let cur_primitive_idx = vertex_buffers.len();
                    {
                        let positions_vec = gltf_adapter.read_positions();
                        let uvs_vec = gltf_adapter.read_uvs();
                        let normals_vec = gltf_adapter.read_normals();
                        let tangents_vec = gltf_adapter.read_tangents();
                        let colors = gltf_adapter.read_colors();
                        let indices_vec = gltf_adapter.read_indices();

                        assert!(positions_vec.len() > 0, "No positions found in gltf");
                        assert_eq!(positions_vec.len(), uvs_vec.len());
                        assert_eq!(positions_vec.len(), normals_vec.len());
                        assert_eq!(positions_vec.len(), tangents_vec.len());
                        assert_eq!(positions_vec.len(), colors.len());
                        assert_eq!(positions_vec.len(), indices_vec.len());

                        for primitive_i in 0..positions_vec.len() {
                            let positions = &positions_vec[primitive_i];
                            let uvs = uvs_vec[primitive_i].as_ref();
                            let normals = &normals_vec[primitive_i];
                            let colors = colors[primitive_i].as_ref();

                            let tangents = if uvs.is_none() {
                                &vec![[0.0, 0.0, 0.0, 1.0]; positions.len()]
                            } else if let Some(tangents) = &tangents_vec[primitive_i] {
                                tangents
                            } else {
                                let indices = &indices_vec[primitive_i];
                                &compute_tangent(
                                    positions,
                                    normals,
                                    uvs.as_ref().unwrap(),
                                    &indices,
                                )
                            };

                            let vertex_len = positions.len();
                            let mut vertices = Vec::with_capacity(vertex_len);
                            for vertex_i in 0..vertex_len {
                                let position = positions[vertex_i];
                                let uv = uvs.map_or([-1.0, -1.0], |uv| uv[vertex_i]);
                                let normal = normals[vertex_i];
                                let tangent = tangents[vertex_i];
                                let color =
                                    colors.map_or([1.0, 1.0, 1.0, 1.0], |color| color[vertex_i]);

                                vertices.push(Vertex {
                                    position: Vec3::new(position[0], position[1], position[2]),
                                    uv: Vec2::new(uv[0], uv[1]),
                                    normal: Vec3::new(normal[0], normal[1], normal[2]),
                                    tangent: Vec4::new(
                                        tangent[0], tangent[1], tangent[2], tangent[3],
                                    ),
                                    color: Vec4::new(color[0], color[1], color[2], color[3]),
                                });
                            }
                            let vertex_buffer = common::buffer::Buffer::new(
                                vertices.as_ptr() as *const std::ffi::c_void,
                                std::mem::size_of::<Vertex>() as u64,
                                vertices.len() as u64,
                                vk::BufferUsageFlags::STORAGE_BUFFER
                                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );
                            vertex_buffers.push(vertex_buffer);
                        }
                    }

                    // below is the code for raytracing
                    {
                        // index buffer
                        log::info!("loading index buffer");
                        {
                            let indices_vec = gltf_adapter.read_indices();
                            for primitive_i in 0..indices_vec.len() {
                                let indices = &indices_vec[primitive_i];
                                let index_buffer = common::buffer::Buffer::new(
                                indices.as_ptr() as *const std::ffi::c_void,
                                std::mem::size_of::<Index>() as u64,
                                indices.len() as u64,
                                vk::BufferUsageFlags::STORAGE_BUFFER
                                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );
                                index_buffers.push(index_buffer);
                            }
                        }

                        // geometry
                        log::info!("loading geometry");
                        for primitive_i in cur_primitive_idx..vertex_buffers.len() {
                            // transform buffer
                            {
                                let identity = create_identity_transform_matrix();
                                let transform_buffer = common::buffer::Buffer::new(
                                identity.matrix.as_ptr() as *const std::ffi::c_void,
                                std::mem::size_of_val(&identity.matrix) as u64,
                                1,
                                vk::BufferUsageFlags::STORAGE_BUFFER
                                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );
                                transform_buffers.push(transform_buffer);
                            }

                            let mut geometry_data =
                                vk::AccelerationStructureGeometryDataKHR::default();
                            geometry_data.triangles =
                                vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                        device_address: vertex_buffers[primitive_i]
                                            .device_address(),
                                    })
                                    .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                                    .index_type(vk::IndexType::UINT32)
                                    .index_data(vk::DeviceOrHostAddressConstKHR {
                                        device_address: index_buffers[primitive_i].device_address(),
                                    })
                                    .transform_data(vk::DeviceOrHostAddressConstKHR {
                                        device_address: transform_buffers[primitive_i]
                                            .device_address(),
                                    });

                            let geometries = vec![vk::AccelerationStructureGeometryKHR::default()
                                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                                .geometry(geometry_data)
                                .flags(vk::GeometryFlagsKHR::OPAQUE)];

                            let geometry_node = GeometryNode {
                                vertex_buffer_device_address: vertex_buffers[primitive_i]
                                    .device_address(),
                                index_buffer_device_address: index_buffers[primitive_i]
                                    .device_address(),
                            };
                            geometry_nodes.push(geometry_node);

                            // get size info
                            let acceleration_structure_build_geometry_info =
                                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                                    .flags(
                                        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                                    )
                                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                                    .geometries(&geometries);

                            let mut acceleration_structure_build_sizes_info =
                                vk::AccelerationStructureBuildSizesInfoKHR::default();
                            ensure!(index_buffers[primitive_i].len() % 3 == 0);
                            let num_triangles = index_buffers[primitive_i].len() as u32 / 3;
                            unsafe {
                                additional_functions
                                    .acceleration_structure
                                    .get_acceleration_structure_build_sizes(
                                        vk::AccelerationStructureBuildTypeKHR::default(),
                                        &acceleration_structure_build_geometry_info,
                                        &[num_triangles],
                                        &mut acceleration_structure_build_sizes_info,
                                    );
                            }
                            let bottom_level_acceleration_structure = common::buffer::Buffer::new(
                                std::ptr::null(),
                                acceleration_structure_build_sizes_info.acceleration_structure_size,
                                1,
                                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS |
                                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );
                            bottom_level_acceleration_structures
                                .push(bottom_level_acceleration_structure);

                            let acceleration_structure_create_info =
                                vk::AccelerationStructureCreateInfoKHR::default()
                                    .buffer(
                                        bottom_level_acceleration_structures
                                            .last()
                                            .unwrap()
                                            .vk_buffer(),
                                    )
                                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                                    .size(
                                        acceleration_structure_build_sizes_info
                                            .acceleration_structure_size,
                                    );
                            let handle = unsafe {
                                additional_functions
                                    .acceleration_structure
                                    .create_acceleration_structure(
                                        &acceleration_structure_create_info,
                                        None,
                                    )?
                            };
                            bottom_level_acceleration_structure_handles.push(handle);

                            let scratch_buffer = common::buffer::Buffer::new(
                                std::ptr::null(),
                                acceleration_structure_build_sizes_info.build_scratch_size,
                                1,
                                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );

                            let geometries = vec![vk::AccelerationStructureGeometryKHR::default()
                                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                                .geometry(geometry_data)
                                .flags(vk::GeometryFlagsKHR::OPAQUE)];

                            let acceleration_structure_build_geometry_info =
                                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                                    .flags(
                                        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                                    )
                                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                                    .dst_acceleration_structure(
                                        bottom_level_acceleration_structure_handles
                                            .last()
                                            .unwrap()
                                            .clone(),
                                    )
                                    .geometries(&geometries)
                                    .scratch_data(vk::DeviceOrHostAddressKHR {
                                        device_address: scratch_buffer.device_address(),
                                    });

                            let range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
                                .primitive_count(num_triangles)
                                .primitive_offset(0)
                                .first_vertex(0)
                                .transform_offset(0);
                            let acceleration_structure_build_range_infos = [range_info];

                            let command_buffer = begin_single_time_command(
                                command_pool.vk_command_pool(),
                                ash_device.clone(),
                            );
                            unsafe {
                                additional_functions
                                    .acceleration_structure
                                    .cmd_build_acceleration_structures(
                                        command_buffer,
                                        &[acceleration_structure_build_geometry_info],
                                        &[&acceleration_structure_build_range_infos],
                                    );
                            }
                            end_single_time_command(
                                command_buffer,
                                graphics_compute_queue,
                                command_pool.vk_command_pool(),
                                ash_device.clone(),
                            );

                            let acceleration_device_address = unsafe {
                                additional_functions
                                    .acceleration_structure
                                    .get_acceleration_structure_device_address(
                                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                                            .acceleration_structure(
                                                bottom_level_acceleration_structure_handles
                                                    .last()
                                                    .unwrap()
                                                    .clone(),
                                            ),
                                    )
                            };
                            bottom_level_acceleration_structure_device_addresses
                                .push(acceleration_device_address);
                        }
                    }

                    // meshlet
                    log::info!("loading meshlet");
                    {
                        let positions_vec = gltf_adapter.read_positions();
                        let indices_vec = gltf_adapter.read_indices();

                        for primitive_i in 0..positions_vec.len() {
                            let positions = &positions_vec[primitive_i];
                            let indices = &indices_vec[primitive_i];
                            let meshlet_object = chroma_scene::meshlet::generate_meshlets(
                                positions.clone(),
                                indices.clone(),
                            );
                            let meshlets = meshlet_object.meshlets;

                            let meshlet_buffer = allocate_device_local_buffer(
                                meshlets.as_ptr() as *const std::ffi::c_void,
                                std::mem::size_of::<meshopt::ffi::meshopt_Meshlet>() as u64,
                                meshlets.len() as u64,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                                command_pool.vk_command_pool(),
                                graphics_compute_queue,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );
                            meshlet_buffers.push(meshlet_buffer);

                            let meshlet_vertices = meshlet_object.meshlet_vertices;
                            let meshlet_vertices_buffer = allocate_device_local_buffer(
                                meshlet_vertices.as_ptr() as *const std::ffi::c_void,
                                std::mem::size_of::<u32>() as u64,
                                meshlet_vertices.len() as u64,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                                command_pool.vk_command_pool(),
                                graphics_compute_queue,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );
                            meshlet_vertices_buffers.push(meshlet_vertices_buffer);

                            let meshlet_triangles = meshlet_object.meshlet_triangles;
                            let meshlet_triangle_buffer = allocate_device_local_buffer(
                                meshlet_triangles.as_ptr() as *const std::ffi::c_void,
                                std::mem::size_of::<u8>() as u64,
                                meshlet_triangles.len() as u64,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                                command_pool.vk_command_pool(),
                                graphics_compute_queue,
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            );
                            meshlet_triangle_buffers.push(meshlet_triangle_buffer);
                        }
                    }

                    // base color images
                    log::info!("loading base color images");
                    {
                        let (mut images, factors) = gltf_adapter.read_base_colors();
                        let mut textures = Vec::with_capacity(images.len());
                        for image in images.iter_mut() {
                            let mut image = image
                                .clone()
                                .unwrap_or(chroma_scene::image::create_white_image(1, 1));
                            image.add_alpha_if_not_exist();
                            let width = image.width;
                            let height = image.height;
                            let texture = Texture::new(
                                vec![image],
                                vk::ImageCreateInfo::default()
                                    .image_type(vk::ImageType::TYPE_2D)
                                    .extent(
                                        vk::Extent3D::default()
                                            .width(width)
                                            .height(height)
                                            .depth(1),
                                    )
                                    .format(vk::Format::R8G8B8A8_SRGB)
                                    .usage(
                                        vk::ImageUsageFlags::TRANSFER_SRC
                                            | vk::ImageUsageFlags::TRANSFER_DST
                                            | vk::ImageUsageFlags::SAMPLED,
                                    )
                                    .tiling(vk::ImageTiling::OPTIMAL)
                                    .initial_layout(vk::ImageLayout::UNDEFINED)
                                    .mip_levels(MIP_LEVEL)
                                    .samples(vk::SampleCountFlags::TYPE_1)
                                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                    .array_layers(1),
                                vk::ImageViewType::TYPE_2D,
                                graphics_compute_queue,
                                command_pool.vk_command_pool(),
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            )?;
                            textures.push(texture);
                        }
                        base_color_textures.extend(textures);
                        base_color_factors.extend(factors);
                    }

                    // normal images
                    log::info!("loading normal images");
                    {
                        let images = gltf_adapter.read_normal_images();
                        let mut textures = Vec::with_capacity(images.len());
                        for image in images.iter() {
                            let mut image = image
                                .clone()
                                .unwrap_or(chroma_scene::image::create_white_image(1, 1));
                            image.add_alpha_if_not_exist();

                            let width = image.width;
                            let height = image.height;
                            let texture = Texture::new(
                                vec![image],
                                vk::ImageCreateInfo::default()
                                    .image_type(vk::ImageType::TYPE_2D)
                                    .extent(
                                        vk::Extent3D::default()
                                            .width(width)
                                            .height(height)
                                            .depth(1),
                                    )
                                    .format(vk::Format::R8G8B8A8_UNORM)
                                    .usage(
                                        vk::ImageUsageFlags::TRANSFER_DST
                                            | vk::ImageUsageFlags::SAMPLED,
                                    )
                                    .tiling(vk::ImageTiling::OPTIMAL)
                                    .initial_layout(vk::ImageLayout::UNDEFINED)
                                    .mip_levels(1)
                                    .samples(vk::SampleCountFlags::TYPE_1)
                                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                    .array_layers(1),
                                vk::ImageViewType::TYPE_2D,
                                graphics_compute_queue,
                                command_pool.vk_command_pool(),
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            )?;
                            textures.push(texture);
                        }
                        normal_textures.extend(textures);
                    }

                    // emissive images
                    log::info!("loading emissive images");
                    {
                        let images = gltf_adapter.read_emissive_images();
                        let mut textures = Vec::with_capacity(images.len());
                        for image in images.iter() {
                            let mut image = image
                                .clone()
                                .unwrap_or(chroma_scene::image::create_black_image(1, 1));
                            image.add_alpha_if_not_exist();

                            let width = image.width;
                            let height = image.height;
                            let texture = Texture::new(
                                vec![image],
                                vk::ImageCreateInfo::default()
                                    .image_type(vk::ImageType::TYPE_2D)
                                    .extent(
                                        vk::Extent3D::default()
                                            .width(width)
                                            .height(height)
                                            .depth(1),
                                    )
                                    .format(vk::Format::R8G8B8A8_SRGB)
                                    .usage(
                                        vk::ImageUsageFlags::TRANSFER_DST
                                            | vk::ImageUsageFlags::SAMPLED,
                                    )
                                    .tiling(vk::ImageTiling::OPTIMAL)
                                    .initial_layout(vk::ImageLayout::UNDEFINED)
                                    .mip_levels(1)
                                    .samples(vk::SampleCountFlags::TYPE_1)
                                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                    .array_layers(1),
                                vk::ImageViewType::TYPE_2D,
                                graphics_compute_queue,
                                command_pool.vk_command_pool(),
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            )?;
                            textures.push(texture);
                        }
                        emissive_textures.extend(textures);
                    }

                    // transmission images
                    log::info!("loading transmission images");
                    {
                        let (mut images, factors) = gltf_adapter.read_transmission();
                        let mut textures = Vec::with_capacity(images.len());
                        for image in images.iter_mut() {
                            let image = image
                                .clone()
                                .unwrap_or(chroma_scene::image::create_white_image(1, 1));
                            let width = image.width;
                            let height = image.height;
                            let texture = Texture::new(
                                vec![image],
                                vk::ImageCreateInfo::default()
                                    .image_type(vk::ImageType::TYPE_2D)
                                    .extent(
                                        vk::Extent3D::default()
                                            .width(width)
                                            .height(height)
                                            .depth(1),
                                    )
                                    .format(vk::Format::R8G8B8A8_UNORM)
                                    .usage(
                                        vk::ImageUsageFlags::TRANSFER_DST
                                            | vk::ImageUsageFlags::SAMPLED,
                                    )
                                    .tiling(vk::ImageTiling::OPTIMAL)
                                    .initial_layout(vk::ImageLayout::UNDEFINED)
                                    .mip_levels(1)
                                    .samples(vk::SampleCountFlags::TYPE_1)
                                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                    .array_layers(1),
                                vk::ImageViewType::TYPE_2D,
                                graphics_compute_queue,
                                command_pool.vk_command_pool(),
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            )?;
                            textures.push(texture);
                        }
                        transmission_textures.extend(textures);
                        transmission_factors.extend(factors);
                    }

                    // metallic_roughness images
                    log::info!("loading metallic_roughness images");
                    {
                        let (images, metallic_factors_vec, roughness_factors_vec) =
                            gltf_adapter.read_metallic_roughnesses();
                        let mut textures = Vec::with_capacity(images.len());
                        for image in images.iter() {
                            let mut image = image
                                .clone()
                                .unwrap_or(chroma_scene::image::create_white_image(1, 1));
                            image.add_alpha_if_not_exist();

                            let width = image.width;
                            let height = image.height;
                            let texture = Texture::new(
                                vec![image],
                                vk::ImageCreateInfo::default()
                                    .image_type(vk::ImageType::TYPE_2D)
                                    .extent(
                                        vk::Extent3D::default()
                                            .width(width)
                                            .height(height)
                                            .depth(1),
                                    )
                                    .format(vk::Format::R8G8B8A8_UNORM)
                                    .usage(
                                        vk::ImageUsageFlags::TRANSFER_DST
                                            | vk::ImageUsageFlags::SAMPLED,
                                    )
                                    .tiling(vk::ImageTiling::OPTIMAL)
                                    .initial_layout(vk::ImageLayout::UNDEFINED)
                                    .mip_levels(1)
                                    .samples(vk::SampleCountFlags::TYPE_1)
                                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                    .array_layers(1),
                                vk::ImageViewType::TYPE_2D,
                                graphics_compute_queue,
                                command_pool.vk_command_pool(),
                                physical_device,
                                ash_device.clone(),
                                instance.clone(),
                            )?;
                            textures.push(texture);
                        }
                        metallic_roughness_textures.extend(textures);

                        // fill in the factors
                        let mut factor_i = 0;
                        for image in images.iter() {
                            if image.is_some() {
                                metallic_factors.push(metallic_factors_vec[factor_i]);
                                roughness_factors.push(roughness_factors_vec[factor_i]);
                                factor_i += 1;
                            } else {
                                metallic_factors.push(0.0);
                                roughness_factors.push(0.0);
                            }
                        }
                    }
                }
            }
        }
        let geometry_node_buffer = common::buffer::Buffer::new(
            geometry_nodes.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of::<GeometryNode>() as u64,
            geometry_nodes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            physical_device,
            ash_device.clone(),
            instance.clone(),
        );

        log::info!("creating top-level acceleration structure");
        let (
            top_level_acceleration_structure,
            top_level_acceleration_structure_handle,
            _top_level_acceleration_structure_device_address,
        ) = {
            let mut instances = Vec::new();
            for address in &bottom_level_acceleration_structure_device_addresses {
                let instance = vk::AccelerationStructureInstanceKHR {
                    transform: create_identity_transform_matrix(),
                    instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
                    instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                        0,
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: address.clone(),
                    },
                };
                instances.push(instance);
            }

            let instance_buffer = common::buffer::Buffer::new(
                instances.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
                instances.len() as u64,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                physical_device,
                ash_device.clone(),
                instance.clone(),
            );

            let mut geo = vk::AccelerationStructureGeometryDataKHR::default();
            geo.instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: instance_buffer.device_address(),
                });

            let acceleration_structure_geometries =
                vec![vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                    .geometry(geo)
                    .flags(vk::GeometryFlagsKHR::OPAQUE)];

            let acceleration_structure_build_geometry_info =
                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .geometries(&acceleration_structure_geometries);

            let mut acceleration_structure_build_sizes_info =
                vk::AccelerationStructureBuildSizesInfoKHR::default();
            let primitive_count = instances.len() as u32;
            unsafe {
                additional_functions
                    .acceleration_structure
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::default(),
                        &acceleration_structure_build_geometry_info,
                        &[primitive_count],
                        &mut acceleration_structure_build_sizes_info,
                    );
            }

            let acceleration_structure = common::buffer::Buffer::new(
                std::ptr::null(),
                acceleration_structure_build_sizes_info.acceleration_structure_size,
                1,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                physical_device,
                ash_device.clone(),
                instance.clone(),
            );

            let acceleration_structure_create_info =
                vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(acceleration_structure.vk_buffer())
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .size(acceleration_structure_build_sizes_info.acceleration_structure_size);

            let handle = unsafe {
                additional_functions
                    .acceleration_structure
                    .create_acceleration_structure(&acceleration_structure_create_info, None)?
            };

            let scratch_buffer = common::buffer::Buffer::new(
                std::ptr::null(),
                acceleration_structure_build_sizes_info.build_scratch_size,
                1,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                physical_device,
                ash_device.clone(),
                instance.clone(),
            );

            let acceleration_structure_geometries =
                vec![vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                            .array_of_pointers(false)
                            .data(vk::DeviceOrHostAddressConstKHR {
                                device_address: instance_buffer.device_address(),
                            }),
                    })];

            let acceleration_structure_build_geometry_info =
                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(handle)
                    .geometries(&acceleration_structure_geometries)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffer.device_address(),
                    });

            let range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

            let acceleration_structure_build_range_infos = [range_info];

            let command_buffer =
                begin_single_time_command(command_pool.vk_command_pool(), ash_device.clone());

            unsafe {
                additional_functions
                    .acceleration_structure
                    .cmd_build_acceleration_structures(
                        command_buffer,
                        &[acceleration_structure_build_geometry_info],
                        &[&acceleration_structure_build_range_infos],
                    );
            }

            end_single_time_command(
                command_buffer,
                graphics_compute_queue,
                command_pool.vk_command_pool(),
                ash_device.clone(),
            );

            let device_address = unsafe {
                additional_functions
                    .acceleration_structure
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(handle),
                    )
            };
            (acceleration_structure, handle, device_address)
        };

        let primitive_size = base_color_textures.len();

        log::info!("creating material uniform buffers");
        let mut material_ubo = Vec::new();
        for primitive_i in 0..primitive_size {
            let material_params = MaterialParams {
                base_color_factor: Vec4::from_row_slice(&base_color_factors[primitive_i]),
                metallic_roughness_transmission_factor: Vec4::new(
                    metallic_factors[primitive_i],
                    roughness_factors[primitive_i],
                    0.0,
                    0.0,
                ),
                ..Default::default()
            };
            let ubo = uniform_buffer::UniformBuffer::<MaterialParams>::new(
                material_params,
                physical_device,
                ash_device.clone(),
                instance.clone(),
            );
            material_ubo.push(ubo);
        }

        // skybox resources
        let skybox_resources = if let Some(skybox) = scene.skybox {
            let width = skybox[&CubeFace::Top].width;
            let height = skybox[&CubeFace::Top].height;
            let skybox_texture = Texture::new(
                vec![
                    skybox[&CubeFace::Front].clone(),
                    skybox[&CubeFace::Back].clone(),
                    skybox[&CubeFace::Top].clone(),
                    skybox[&CubeFace::Bottom].clone(),
                    skybox[&CubeFace::Left].clone(),
                    skybox[&CubeFace::Right].clone(),
                ],
                vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D::default().width(width).height(height).depth(1))
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .mip_levels(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .array_layers(skybox.len() as u32)
                    .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE),
                vk::ImageViewType::CUBE,
                graphics_compute_queue,
                command_pool.vk_command_pool(),
                physical_device,
                ash_device.clone(),
                instance.clone(),
            )?;
            Some(SkyboxResources { skybox_texture })
        } else {
            None
        };

        log::info!("creating render targets");
        let graphics_render_targets = {
            let mut render_targets = HashMap::new();
            // output
            render_targets.insert(
                GraphicsRenderTarget::Output,
                ImageBuffer::new(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(swapchain.vk_swapchain_format())
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.vk_swapchain_extent().width)
                                .height(swapchain.vk_swapchain_extent().height)
                                .depth(1),
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::STORAGE,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .array_layers(1)
                        .mip_levels(1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    vk::ImageViewType::TYPE_2D,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::ImageAspectFlags::COLOR,
                    physical_device,
                    ash_device.clone(),
                    instance.clone(),
                )?,
            );
            // depth
            render_targets.insert(
                GraphicsRenderTarget::DepthStencil,
                ImageBuffer::new(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::D32_SFLOAT)
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.vk_swapchain_extent().width)
                                .height(swapchain.vk_swapchain_extent().height)
                                .depth(1),
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::STORAGE,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .array_layers(1)
                        .mip_levels(1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    vk::ImageViewType::TYPE_2D,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::ImageAspectFlags::DEPTH,
                    physical_device,
                    ash_device.clone(),
                    instance.clone(),
                )?,
            );
            // color
            render_targets.insert(
                GraphicsRenderTarget::Color,
                ImageBuffer::new(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.vk_swapchain_extent().width)
                                .height(swapchain.vk_swapchain_extent().height)
                                .depth(1),
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .array_layers(1)
                        .mip_levels(1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    vk::ImageViewType::TYPE_2D,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::ImageAspectFlags::COLOR,
                    physical_device,
                    ash_device.clone(),
                    instance.clone(),
                )?,
            );
            // normal
            render_targets.insert(
                GraphicsRenderTarget::Normal,
                ImageBuffer::new(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.vk_swapchain_extent().width)
                                .height(swapchain.vk_swapchain_extent().height)
                                .depth(1),
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .array_layers(1)
                        .mip_levels(1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    vk::ImageViewType::TYPE_2D,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::ImageAspectFlags::COLOR,
                    physical_device,
                    ash_device.clone(),
                    instance.clone(),
                )?,
            );
            // metallic_roughness
            render_targets.insert(
                GraphicsRenderTarget::MetallicRoughness,
                ImageBuffer::new(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.vk_swapchain_extent().width)
                                .height(swapchain.vk_swapchain_extent().height)
                                .depth(1),
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .array_layers(1)
                        .mip_levels(1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    vk::ImageViewType::TYPE_2D,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::ImageAspectFlags::COLOR,
                    physical_device,
                    ash_device.clone(),
                    instance.clone(),
                )?,
            );

            // emissive
            render_targets.insert(
                GraphicsRenderTarget::Emissive,
                ImageBuffer::new(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_SRGB)
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.vk_swapchain_extent().width)
                                .height(swapchain.vk_swapchain_extent().height)
                                .depth(1),
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .array_layers(1)
                        .mip_levels(1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    vk::ImageViewType::TYPE_2D,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::ImageAspectFlags::COLOR,
                    physical_device,
                    ash_device.clone(),
                    instance.clone(),
                )?,
            );
            render_targets
        };

        let shadow_render_targets = {
            let mut render_targets = HashMap::new();
            // output
            render_targets.insert(
                ShadowRenderTarget::Output,
                ImageBuffer::new(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.vk_swapchain_extent().width)
                                .height(swapchain.vk_swapchain_extent().height)
                                .depth(1),
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .array_layers(1)
                        .mip_levels(1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    vk::ImageViewType::TYPE_2D,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::ImageAspectFlags::COLOR,
                    physical_device,
                    ash_device.clone(),
                    instance.clone(),
                )?,
            );
            render_targets
        };

        log::info!("creating graphics descriptor pool");
        let graphics_descriptor_pool = descriptor_pool::DescriptorPool::new(
            vk::DescriptorPoolCreateInfo::default()
                .max_sets(
                    MAX_FRAMES_IN_FLIGHT as u32
                        * primitive_size as u32
                        * NUM_GRAPHICS_DESCRIPTOR_SETS as u32,
                )
                .pool_sizes(&[
                    // transform + material
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * primitive_size as u32 * 2),
                    // meshlet + meshlet vertices + meshlet triangles + vertex inputs
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * primitive_size as u32 * 4),
                    // color + normal + metallic_roughness + emissive
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * primitive_size as u32 * 4),
                ])
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
            ash_device.clone(),
        );

        log::info!("creating graphics descriptor set layout");
        let graphics_descriptor_set_layouts = vec![
            // set 0
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // transform
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::MESH_EXT),
                    // color
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    // normal
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    // metallic roughness
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(3)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    // emissive
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(4)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                ]),
                ash_device.clone(),
            ),
            // set 1
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // material
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    // meshlet
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::MESH_EXT),
                    // meshlet vertices
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::MESH_EXT),
                    // meshlet triangles
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::MESH_EXT),
                    // vertex inputs
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(4)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::MESH_EXT),
                ]),
                ash_device.clone(),
            ),
        ];

        log::info!("creating graphics descriptor sets");
        let mut graphics_descriptor_sets = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let mut layouts = Vec::new();
            let layout = graphics_descriptor_set_layouts
                .iter()
                .map(|layout| layout.vk_descriptor_set_layout())
                .collect::<Vec<_>>();
            for _ in 0..primitive_size {
                layouts.extend(layout.clone());
            }
            let descriptor_set = common::descriptor_set::DescriptorSet::new(
                vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(graphics_descriptor_pool.vk_pool())
                    .set_layouts(&layouts),
                graphics_descriptor_pool.vk_pool(),
                ash_device.clone(),
            );
            graphics_descriptor_sets.push(descriptor_set);
        }

        log::info!("updating graphics descriptor sets");
        for frame_i in 0..MAX_FRAMES_IN_FLIGHT {
            let transform_ubo_buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(transform_ubo.vk_buffer(frame_i))
                .offset(0)
                .range(transform_ubo.type_size())];

            for (primitive_i, set_i) in (0..primitive_size).zip(
                (0..primitive_size * NUM_GRAPHICS_DESCRIPTOR_SETS)
                    .step_by(NUM_GRAPHICS_DESCRIPTOR_SETS),
            ) {
                let mut descriptor_writes = Vec::new();
                // transform
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i))
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&transform_ubo_buffer_info),
                );

                // color
                let base_color_image_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(base_color_textures[primitive_i].image_view())
                    .sampler(create_texture_sampler(
                        MIP_LEVEL,
                        physical_device,
                        ash_device.clone(),
                        &instance,
                        &entry,
                    ))];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i))
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&base_color_image_info),
                );

                // normal
                let normal_image_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(normal_textures[primitive_i].image_view())
                    .sampler(create_texture_sampler(
                        1,
                        physical_device,
                        ash_device.clone(),
                        &instance,
                        &entry,
                    ))];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i))
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&normal_image_info),
                );

                // metallic_roughness
                let metallic_roughness_image_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(metallic_roughness_textures[primitive_i].image_view())
                    .sampler(create_texture_sampler(
                        1,
                        physical_device,
                        ash_device.clone(),
                        &instance,
                        &entry,
                    ))];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i))
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&metallic_roughness_image_info),
                );

                // emissive
                let emissive_image_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(emissive_textures[primitive_i].image_view())
                    .sampler(create_texture_sampler(
                        1,
                        physical_device,
                        ash_device.clone(),
                        &instance,
                        &entry,
                    ))];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i))
                        .dst_binding(4)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&emissive_image_info),
                );

                // material
                let material_ubo_buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(material_ubo[primitive_i].vk_buffer(frame_i))
                    .offset(0)
                    .range(material_ubo[primitive_i].type_size())];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(
                            graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i + 1),  // Need to +1 for descriptor set 1
                        )
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&material_ubo_buffer_info),
                );

                // meshlet
                let meshlet_buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(meshlet_buffers[primitive_i].vk_buffer())
                    .offset(0)
                    .range(meshlet_buffers[primitive_i].size() as u64)];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(
                            graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i + 1),  // Need to +1 for descriptor set 1
                        )
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&meshlet_buffer_info),
                );

                // meshlet vertices
                let meshlet_vertices_buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(meshlet_vertices_buffers[primitive_i].vk_buffer())
                    .offset(0)
                    .range(meshlet_vertices_buffers[primitive_i].size() as u64)];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(
                            graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i + 1),  // Need to +1 for descriptor set 1
                        )
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&meshlet_vertices_buffer_info),
                );

                // meshlet triangles
                let meshlet_triangle_buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(meshlet_triangle_buffers[primitive_i].vk_buffer())
                    .offset(0)
                    .range(meshlet_triangle_buffers[primitive_i].size() as u64)];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(
                            graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i + 1),  // Need to +1 for descriptor set 1
                        )
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&meshlet_triangle_buffer_info),
                );

                // vertex inputs
                let vertex_input_buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(vertex_buffers[primitive_i].vk_buffer())
                    .offset(0)
                    .range(
                        std::mem::size_of::<Vertex>() as u64
                            * vertex_buffers[primitive_i].len() as u64,
                    )];
                descriptor_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(
                            graphics_descriptor_sets[frame_i].vk_descriptor_set(set_i + 1),  // Need to +1 for descriptor set 1
                        )
                        .dst_binding(4)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&vertex_input_buffer_info),
                );

                unsafe {
                    ash_device.update_descriptor_sets(&descriptor_writes, &[]);
                }
            }
        }

        log::info!("creating render pass");
        let graphics_render_pass = {
            let render_pass_attachments = [
                // color
                vk::AttachmentDescription::default()
                    .format(
                        graphics_render_targets
                            .get(&GraphicsRenderTarget::Color)
                            .unwrap()
                            .format(),
                    )
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                // normal
                vk::AttachmentDescription::default()
                    .format(
                        graphics_render_targets
                            .get(&GraphicsRenderTarget::Normal)
                            .unwrap()
                            .format(),
                    )
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                // metallic_roughness
                vk::AttachmentDescription::default()
                    .format(
                        graphics_render_targets
                            .get(&GraphicsRenderTarget::MetallicRoughness)
                            .unwrap()
                            .format(),
                    )
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                // emissive
                vk::AttachmentDescription::default()
                    .format(
                        graphics_render_targets
                            .get(&GraphicsRenderTarget::Emissive)
                            .unwrap()
                            .format(),
                    )
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                // depth
                vk::AttachmentDescription::default()
                    .format(
                        graphics_render_targets
                            .get(&GraphicsRenderTarget::DepthStencil)
                            .unwrap()
                            .format(),
                    )
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            ];

            let color_attachments = [
                // color
                vk::AttachmentReference::default()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                // normal
                vk::AttachmentReference::default()
                    .attachment(1)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                // metallic_roughness
                vk::AttachmentReference::default()
                    .attachment(2)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                // emissive
                vk::AttachmentReference::default()
                    .attachment(3)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            ];

            let depth_stencil_attachment = vk::AttachmentReference::default()
                .attachment(4)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachments)
                .depth_stencil_attachment(&depth_stencil_attachment)];

            let renderpass_create_info = vk::RenderPassCreateInfo::default()
                .attachments(&render_pass_attachments)
                .subpasses(&subpasses);

            common::render_pass::RenderPass::new(renderpass_create_info, ash_device.clone())?
        };

        log::info!("creating graphics pipeline layout");
        let mut graphics_pipeline_layouts = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let layouts = graphics_descriptor_set_layouts
                .iter()
                .map(|layout| layout.vk_descriptor_set_layout())
                .collect::<Vec<_>>();
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

            let pipeline_layout = pipeline_layout::PipelineLayout::new(
                pipeline_layout_create_info,
                ash_device.clone(),
            )?;
            graphics_pipeline_layouts.push(pipeline_layout);
        }

        log::info!("creating graphics pipeline");
        let mut graphics_pipelines = Vec::new();
        for frame_i in 0..MAX_FRAMES_IN_FLIGHT {
            let spv_root = get_shader_spv_root()?;
            let task_shader_code = read_shader_code(&spv_root.join("deferred/shader.task.spv"))?;
            let mesh_shader_code = read_shader_code(&spv_root.join("deferred/shader.mesh.spv"))?;
            let frag_shader_code = read_shader_code(&spv_root.join("deferred/shader.frag.spv"))?;
            let task_shader_module = create_shader_module(&ash_device, task_shader_code)?;
            let mesh_shader_module = create_shader_module(&ash_device, mesh_shader_code)?;
            let frag_shader_module = create_shader_module(&ash_device, frag_shader_code)?;

            let main_function_name = CString::new("main")?;

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .module(task_shader_module)
                    .stage(vk::ShaderStageFlags::TASK_EXT)
                    .name(&main_function_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .module(mesh_shader_module)
                    .stage(vk::ShaderStageFlags::MESH_EXT)
                    .name(&main_function_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .module(frag_shader_module)
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .name(&main_function_name),
            ];

            let binding_descriptions = Vertex::get_binding_descriptions();
            let attribute_descriptions = Vertex::get_attribute_descriptions();
            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_descriptions)
                .vertex_attribute_descriptions(&attribute_descriptions);

            let vertex_input_assembly_state_info =
                vk::PipelineInputAssemblyStateCreateInfo::default()
                    .primitive_restart_enable(false)
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let rasterization_state_create_info =
                vk::PipelineRasterizationStateCreateInfo::default()
                    .depth_clamp_enable(false)
                    .cull_mode(vk::CullModeFlags::BACK)
                    .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                    .line_width(1.0)
                    .polygon_mode(vk::PolygonMode::FILL)
                    .depth_bias_enable(false);

            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .sample_shading_enable(false);

            let depth_stencil_state_create_info =
                vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_compare_op(vk::CompareOp::LESS)
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_bounds_test_enable(false)
                    .stencil_test_enable(false);

            let color_blend_attachment_states = [
                // color
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                    .alpha_blend_op(vk::BlendOp::ADD),
                // normal
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                    .alpha_blend_op(vk::BlendOp::ADD),
                // metallic_roughness
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                    .alpha_blend_op(vk::BlendOp::ADD),
                // emissive
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                    .alpha_blend_op(vk::BlendOp::ADD),
            ];

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(&color_blend_attachment_states)
                .blend_constants([0.0, 0.0, 0.0, 0.0]);

            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state_create_info)
                .input_assembly_state(&vertex_input_assembly_state_info)
                .rasterization_state(&rasterization_state_create_info)
                .multisample_state(&multisample_state_create_info)
                .depth_stencil_state(&depth_stencil_state_create_info)
                .color_blend_state(&color_blend_state)
                .layout(graphics_pipeline_layouts[frame_i].vk_pipeline_layout())
                .render_pass(graphics_render_pass.vk_render_pass())
                .subpass(0)
                .dynamic_state(&dynamic_state)
                .viewport_state(&viewport_state)];

            let graphics_pipeline = common::graphics_pipeline::GraphicsPipeline::new(
                &graphic_pipeline_create_infos,
                ash_device.clone(),
            )?;

            unsafe {
                ash_device.destroy_shader_module(mesh_shader_module, None);
                ash_device.destroy_shader_module(frag_shader_module, None);
            }
            graphics_pipelines.push(graphics_pipeline);
        }

        let mut graphics_framebuffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_views = [
                graphics_render_targets
                    .get(&GraphicsRenderTarget::Color)
                    .unwrap()
                    .image_view(),
                graphics_render_targets
                    .get(&GraphicsRenderTarget::Normal)
                    .unwrap()
                    .image_view(),
                graphics_render_targets
                    .get(&GraphicsRenderTarget::MetallicRoughness)
                    .unwrap()
                    .image_view(),
                graphics_render_targets
                    .get(&GraphicsRenderTarget::Emissive)
                    .unwrap()
                    .image_view(),
                graphics_render_targets
                    .get(&GraphicsRenderTarget::DepthStencil)
                    .unwrap()
                    .image_view(),
            ];
            let output_framebuffer = common::framebuffer::Framebuffer::new(
                graphics_render_pass.vk_render_pass(),
                &image_views,
                swapchain.vk_swapchain_extent().width,
                swapchain.vk_swapchain_extent().height,
                ash_device.clone(),
            );
            graphics_framebuffers.push(output_framebuffer);
        }

        log::info!("creating compute descriptor set layout");
        let compute_descriptor_set_layouts = vec![
            // set 0
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // transform
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    // camera
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                ]),
                ash_device.clone(),
            ),
            // set 1
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // output
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    // color
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    // normal
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    // depth
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    // metallic_roughness
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(4)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    // emissive
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(5)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    // shadow
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(6)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                ]),
                ash_device.clone(),
            ),
            // set 2
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // skybox
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                ]),
                ash_device.clone(),
            ),
        ];

        log::info!("creating compute descriptor pool");
        let compute_descriptor_pool = descriptor_pool::DescriptorPool::new(
            vk::DescriptorPoolCreateInfo::default()
                .max_sets((MAX_FRAMES_IN_FLIGHT as u32) * (NUM_COMPUTE_DESCRIPTOR_SETS as u32))
                .pool_sizes(&[
                    // transform + camera
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * 2),
                    // depth
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                    // color + normal + output + metallic_roughness + emissive + shadow
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count((MAX_FRAMES_IN_FLIGHT as u32) * 6),
                    // skybox
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                ])
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
            ash_device.clone(),
        );

        log::info!("creating compute descriptor sets");
        let mut compute_descriptor_sets = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let layouts = compute_descriptor_set_layouts
                .iter()
                .map(|layout| layout.vk_descriptor_set_layout())
                .collect::<Vec<_>>();
            let descriptor_set = common::descriptor_set::DescriptorSet::new(
                vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(compute_descriptor_pool.vk_pool())
                    .set_layouts(&layouts),
                compute_descriptor_pool.vk_pool(),
                ash_device.clone(),
            );
            compute_descriptor_sets.push(descriptor_set);
        }

        log::info!("updating compute descriptor sets");
        for frame_i in 0..MAX_FRAMES_IN_FLIGHT {
            let mut descriptor_writes = Vec::new();

            // transform
            let transform_ubo_buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(transform_ubo.vk_buffer(frame_i))
                .offset(0)
                .range(transform_ubo.type_size())];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(0))
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&transform_ubo_buffer_info),
            );

            // camera
            let camera_ubo_buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(camera_ubo.vk_buffer(frame_i))
                .offset(0)
                .range(camera_ubo.type_size())];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(0))
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&camera_ubo_buffer_info),
            );

            // output
            let output_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    graphics_render_targets
                        .get(&GraphicsRenderTarget::Output)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&output_image_info),
            );

            // color
            let color_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    graphics_render_targets
                        .get(&GraphicsRenderTarget::Color)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&color_image_info),
            );

            // normal
            let normal_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    graphics_render_targets
                        .get(&GraphicsRenderTarget::Normal)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&normal_image_info),
            );

            // depth
            let depth_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    graphics_render_targets
                        .get(&GraphicsRenderTarget::DepthStencil)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&depth_image_info),
            );

            // metallic_roughness
            let metallic_roughness_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    graphics_render_targets
                        .get(&GraphicsRenderTarget::MetallicRoughness)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(4)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&metallic_roughness_image_info),
            );

            // emissive
            let emissive_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    graphics_render_targets
                        .get(&GraphicsRenderTarget::Emissive)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(5)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&emissive_image_info),
            );

            // shadow
            let shadow_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    shadow_render_targets
                        .get(&ShadowRenderTarget::Output)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(6)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&shadow_image_info),
            );

            // skybox
            let skybox_image_info = if let Some(skybox_resources) = &skybox_resources {
                [vk::DescriptorImageInfo::default()
                    .image_view(skybox_resources.skybox_texture.image_view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(create_texture_sampler(
                        1,
                        physical_device,
                        ash_device.clone(),
                        &instance,
                        &entry,
                    ))]
            } else {
                panic!("skybox resources not found");
            };
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(compute_descriptor_sets[frame_i].vk_descriptor_set(2))
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&skybox_image_info),
            );

            unsafe {
                ash_device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        log::info!("creating compute pipeline layout");
        let mut compute_pipeline_layouts = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let layouts = compute_descriptor_set_layouts
                .iter()
                .map(|layout| layout.vk_descriptor_set_layout())
                .collect::<Vec<_>>();
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

            let pipeline_layout = pipeline_layout::PipelineLayout::new(
                pipeline_layout_create_info,
                ash_device.clone(),
            )?;
            compute_pipeline_layouts.push(pipeline_layout);
        }

        let mut compute_pipelines = Vec::new();
        for frame_i in 0..MAX_FRAMES_IN_FLIGHT {
            let spv_root = get_shader_spv_root()?;
            let compute_shader_code = read_shader_code(&spv_root.join("deferred/shader.comp.spv"))?;
            let compute_shader_module = create_shader_module(&ash_device, compute_shader_code)?;
            let main_function_name = CString::new("main")?;

            let shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .module(compute_shader_module)
                .stage(vk::ShaderStageFlags::COMPUTE)
                .name(&main_function_name);

            let compute_pipeline_create_infos = [vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage)
                .layout(compute_pipeline_layouts[frame_i].vk_pipeline_layout())];

            let compute_pipeline = common::compute_pipeline::ComputePipeline::new(
                &compute_pipeline_create_infos,
                ash_device.clone(),
            )?;

            unsafe {
                ash_device.destroy_shader_module(compute_shader_module, None);
            }
            compute_pipelines.push(compute_pipeline);
        }

        log::info!("creating shadow descriptor set layout");
        let shadow_descriptor_set_layouts = vec![
            // set 0
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // TLAS
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .stage_flags(
                            vk::ShaderStageFlags::RAYGEN_KHR
                                | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        ),
                    // transform
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(
                            vk::ShaderStageFlags::RAYGEN_KHR
                                | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                                | vk::ShaderStageFlags::MISS_KHR,
                        ),
                    // TODO: light
                ]),
                ash_device.clone(),
            ),
            // set 1
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // output
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                    // skybox
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(
                            vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::MISS_KHR,
                        ),
                ]),
                ash_device.clone(),
            ),
            // set 2
            common::descriptor_set_layout::DescriptorSetLayout::new(
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    // geometry nodes
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(
                            vk::ShaderStageFlags::RAYGEN_KHR
                                | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        ),
                ]),
                ash_device.clone(),
            ),
        ];

        log::info!("creating shadow descriptor pool");
        let shadow_descriptor_pool = descriptor_pool::DescriptorPool::new(
            vk::DescriptorPoolCreateInfo::default()
                .max_sets((MAX_FRAMES_IN_FLIGHT as u32) * (NUM_SHADOW_DESCRIPTOR_SETS as u32))
                .pool_sizes(&[
                    // TLAS
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                    // output
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                    // transform
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                    // skybox
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                    // geometry nodes
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                ])
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
            ash_device.clone(),
        );

        log::info!("creating shadow descriptor sets");
        let mut shadow_descriptor_sets = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let layouts = shadow_descriptor_set_layouts
                .iter()
                .map(|layout| layout.vk_descriptor_set_layout())
                .collect::<Vec<_>>();
            let descriptor_set = common::descriptor_set::DescriptorSet::new(
                vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(shadow_descriptor_pool.vk_pool())
                    .set_layouts(&layouts),
                shadow_descriptor_pool.vk_pool(),
                ash_device.clone(),
            );
            shadow_descriptor_sets.push(descriptor_set);
        }

        log::info!("updating shadow descriptor sets");
        for frame_i in 0..MAX_FRAMES_IN_FLIGHT {
            let mut descriptor_writes = Vec::new();

            // TLAS
            let tlas_handles = [top_level_acceleration_structure_handle];
            let mut tlas_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&tlas_handles);
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(shadow_descriptor_sets[frame_i].vk_descriptor_set(0))
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_count(1) // it contains TLAS
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .push_next(&mut tlas_info),
            );

            // transform
            let transform_ubo_buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(transform_ubo.vk_buffer(frame_i))
                .offset(0)
                .range(transform_ubo.type_size())];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(shadow_descriptor_sets[frame_i].vk_descriptor_set(0))
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&transform_ubo_buffer_info),
            );

            // output
            let output_image_info = [vk::DescriptorImageInfo::default()
                .image_view(
                    shadow_render_targets
                        .get(&ShadowRenderTarget::Output)
                        .unwrap()
                        .image_view(),
                )
                .image_layout(vk::ImageLayout::GENERAL)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(shadow_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&output_image_info),
            );

            // skybox
            let skybox_image_info = if let Some(skybox_resources) = &skybox_resources {
                [vk::DescriptorImageInfo::default()
                    .image_view(skybox_resources.skybox_texture.image_view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(create_texture_sampler(
                        1,
                        physical_device,
                        ash_device.clone(),
                        &instance,
                        &entry,
                    ))]
            } else {
                panic!("skybox resources not found");
            };
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(shadow_descriptor_sets[frame_i].vk_descriptor_set(1))
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&skybox_image_info),
            );

            // geometry nodes
            let geometry_node_buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(geometry_node_buffer.vk_buffer())
                .offset(0)
                .range(geometry_node_buffer.size() as u64)];
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(shadow_descriptor_sets[frame_i].vk_descriptor_set(2))
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&geometry_node_buffer_info),
            );

            unsafe {
                ash_device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        log::info!("creating raytracing pipeline layout");
        let mut shadow_pipeline_layouts = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let layouts = shadow_descriptor_set_layouts
                .iter()
                .map(|layout| layout.vk_descriptor_set_layout())
                .collect::<Vec<_>>();
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

            let pipeline_layout = pipeline_layout::PipelineLayout::new(
                pipeline_layout_create_info,
                ash_device.clone(),
            )?;
            shadow_pipeline_layouts.push(pipeline_layout);
        }

        log::info!("creating raytracing pipeline");
        let mut shadow_pipelines = Vec::new();
        for frame_i in 0..MAX_FRAMES_IN_FLIGHT {
            let spv_root = get_shader_spv_root()?;
            let raygen_shader_code = read_shader_code(&spv_root.join("deferred/shadow.rgen.spv"))?;
            let miss_shader_code = read_shader_code(&spv_root.join("deferred/shadow.rmiss.spv"))?;
            let closest_hit_shader_code =
                read_shader_code(&spv_root.join("deferred/shadow.rchit.spv"))?;
            let raygen_shader_module = create_shader_module(&ash_device, raygen_shader_code)?;
            let miss_shader_module = create_shader_module(&ash_device, miss_shader_code)?;
            let closest_hit_shader_module =
                create_shader_module(&ash_device, closest_hit_shader_code)?;
            let main_function_name = CString::new("main")?;

            let raygen_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .module(raygen_shader_module)
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .name(&main_function_name);
            let miss_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .module(miss_shader_module)
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .name(&main_function_name);
            let closest_hit_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .module(closest_hit_shader_module)
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .name(&main_function_name);

            let shader_stages = vec![
                raygen_shader_stage_create_info,
                miss_shader_stage_create_info,
                closest_hit_shader_stage_create_info,
            ];

            let create_infos = [
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(0)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(1)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(2)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            ];
            let raytracing_pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
                .stages(&shader_stages)
                .groups(&create_infos)
                .layout(shadow_pipeline_layouts[frame_i].vk_pipeline_layout())
                .max_pipeline_ray_recursion_depth(2);

            let shadow_pipeline = common::raytracing_pipeline::RaytracingPipeline::new(
                &raytracing_pipeline_create_info,
                ash_device.clone(),
                additional_functions.raytracing_pipeline.clone(),
            );
            unsafe {
                ash_device.destroy_shader_module(raygen_shader_module, None);
                ash_device.destroy_shader_module(miss_shader_module, None);
                ash_device.destroy_shader_module(closest_hit_shader_module, None);
            }
            shadow_pipelines.push(shadow_pipeline);
        }

        log::info!("get raytracing props");
        let raytracing_pipeline_props = {
            let mut raytracing_pipeline_props =
                PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut physical_device_props =
                vk::PhysicalDeviceProperties2::default().push_next(&mut raytracing_pipeline_props);
            unsafe {
                additional_functions
                    .get_physical_device_properties2
                    .get_physical_device_properties2(physical_device, &mut physical_device_props);
            }
            raytracing_pipeline_props
        };

        log::info!("create shader binding table");
        let handle_size = raytracing_pipeline_props.shader_group_handle_size;
        let handle_size_aligned = math::align_up(
            handle_size,
            raytracing_pipeline_props.shader_group_handle_alignment,
        );
        let group_count = 3; // rgen + miss + chit

        let mut shadow_shader_binding_tables = Vec::new();
        for frame_i in 0..MAX_FRAMES_IN_FLIGHT {
            let shader_handle_storage = unsafe {
                additional_functions
                    .raytracing_pipeline
                    .get_ray_tracing_shader_group_handles(
                        shadow_pipelines[frame_i].pipeline(),
                        0,
                        group_count,
                        (handle_size_aligned * group_count) as usize,
                    )?
            };

            let buffer_usage_flags = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let memory_property_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let raygen_shader_binding_table = Buffer::new(
                shader_handle_storage.as_ptr() as *const _,
                handle_size as u64,
                1,
                buffer_usage_flags,
                memory_property_flags,
                physical_device,
                ash_device.clone(),
                instance.clone(),
            );

            let miss_shader_binding_table = Buffer::new(
                unsafe {
                    shader_handle_storage
                        .as_ptr()
                        .add(handle_size_aligned as usize) as *const _
                },
                handle_size as u64,
                1,
                buffer_usage_flags,
                memory_property_flags,
                physical_device,
                ash_device.clone(),
                instance.clone(),
            );

            let closest_hit_shader_binding_table = Buffer::new(
                unsafe {
                    shader_handle_storage
                        .as_ptr()
                        .add(handle_size_aligned as usize * 2) as *const _
                },
                handle_size as u64,
                1,
                buffer_usage_flags,
                memory_property_flags,
                physical_device,
                ash_device.clone(),
                instance.clone(),
            );

            let binding_table = ShaderBindingTable {
                raygen: raygen_shader_binding_table,
                miss: miss_shader_binding_table,
                hit: closest_hit_shader_binding_table,
            };

            shadow_shader_binding_tables.push(binding_table);
        }

        Ok(Self {
            _vertex_buffers: vertex_buffers,
            _index_buffers: index_buffers,
            _material_ubo: material_ubo,
            ash_device,
            graphics_framebuffers,
            _command_pool: command_pool,
            graphics_render_pass,
            graphics_pipelines,
            graphics_pipeline_layouts,
            _graphics_descriptor_pool: graphics_descriptor_pool,
            graphics_descriptor_sets,
            _compute_descriptor_pool: compute_descriptor_pool,
            compute_descriptor_sets,
            compute_pipelines,
            compute_pipeline_layouts,
            graphics_render_targets,
            meshlet_buffers,
            _meshlet_triangle_buffers: meshlet_triangle_buffers,
            _meshlet_vertices_buffers: meshlet_vertices_buffers,
            _transform_ubo: transform_ubo,
            _camera_ubo: camera_ubo,
            _base_color_textures: base_color_textures,
            _normal_textures: normal_textures,
            _metallic_roughness_textures: metallic_roughness_textures,
            _emissive_textures: emissive_textures,
            _skybox_resources: skybox_resources,
            additional_functions,
            _transform_buffers: transform_buffers,
            shadow_resource: ShadowResource {
                _shadow_descriptor_pool: shadow_descriptor_pool,
                _bottom_level_acceleration_structures: bottom_level_acceleration_structures,
                _bottom_level_acceleration_structure_handles:
                    bottom_level_acceleration_structure_handles,
                _top_level_acceleration_structure: top_level_acceleration_structure,
                _top_level_acceleration_structure_handle: top_level_acceleration_structure_handle,
                shadow_render_targets,
                shadow_descriptor_sets,
                shadow_pipeline_layouts,
                shadow_pipelines,
                shadow_shader_binding_tables,
                _geometry_node_buffer: geometry_node_buffer,
                _geometry_nodes: geometry_nodes,
            },
            shader_group_base_alignment: raytracing_pipeline_props.shader_group_base_alignment,
            shader_group_handle_alignment: raytracing_pipeline_props.shader_group_handle_alignment,
            shader_group_handle_size: raytracing_pipeline_props.shader_group_handle_size,
        })
    }
}

impl DrawStrategy for Deferred {
    fn draw(&self, command_buffer: vk::CommandBuffer, image_index: u32) -> Result<()> {
        let device = &self.ash_device;

        log::debug!("draw deferred rendering");

        let clear_values = [
            // color
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            // normal
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            // metallic_roughness
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            // emissive
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            // depth
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let extent = self
            .graphics_render_targets
            .get(&GraphicsRenderTarget::Output)
            .unwrap()
            .extent();
        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.graphics_render_pass.vk_render_pass())
            .framebuffer(
                self.graphics_framebuffers[image_index as usize]
                    .vk_framebuffer()
                    .clone(),
            )
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }

        assert_eq!(
            self.graphics_pipelines[image_index as usize]
                .vk_pipelines()
                .len(),
            1
        );
        log::debug!("bind pipeline");
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipelines[image_index as usize].vk_pipelines()[0].clone(),
            );
        }

        log::debug!("set viewport and scissor");
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: extent.width,
                height: extent.height,
            },
        }];
        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        unsafe {
            device.cmd_set_scissor(command_buffer, 0, &scissors);
            device.cmd_set_viewport(command_buffer, 0, &viewports);
        }

        log::debug!("draw primitives");
        for primitive_i in 0..self.meshlet_buffers.len() {
            let set_indices = primitive_i * NUM_GRAPHICS_DESCRIPTOR_SETS
                ..(primitive_i + 1) * NUM_GRAPHICS_DESCRIPTOR_SETS;
            let descriptor_sets = set_indices
                .map(|i| self.graphics_descriptor_sets[image_index as usize].vk_descriptor_set(i))
                .collect::<Vec<_>>();
            unsafe {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline_layouts[image_index as usize].vk_pipeline_layout(),
                    0,
                    &descriptor_sets,
                    &[],
                );
            }

            unsafe {
                self.additional_functions
                    .mesh_shader
                    .cmd_draw_mesh_tasks(command_buffer, 1, 1, 1);
            }
        }

        log::debug!("end render pass");
        unsafe {
            device.cmd_end_render_pass(command_buffer);
        }

        {
            log::debug!("color/depth: write -> read");
            let barriers = [
                // color: write -> read
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(PipelineStageFlags2::ALL_GRAPHICS)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(PipelineStageFlags2::ALL_GRAPHICS)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.graphics_render_targets[&GraphicsRenderTarget::Color].image())
                    .subresource_range(ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                // depth: write -> read
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(PipelineStageFlags2::ALL_GRAPHICS)
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(PipelineStageFlags2::ALL_GRAPHICS)
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
                    .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                    .image(
                        self.graphics_render_targets[&GraphicsRenderTarget::DepthStencil].image(),
                    )
                    .subresource_range(ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            ];
            let dependency_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
            unsafe {
                device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
            }
        }

        log::debug!("Draw shadow pass");
        let handle_size = self.shader_group_handle_size;
        let handle_size_aligned = math::align_up(handle_size, self.shader_group_handle_alignment);
        let base_alignment = self.shader_group_base_alignment;

        log::debug!("bind pipeline");
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.shadow_resource.shadow_pipelines[image_index as usize].pipeline(),
            );
        }

        log::debug!("bind descriptor sets");
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.shadow_resource.shadow_pipeline_layouts[image_index as usize]
                    .vk_pipeline_layout(),
                0,
                &self.shadow_resource.shadow_descriptor_sets[image_index as usize]
                    .vk_descriptor_sets(),
                &[],
            );
        }

        log::debug!("trace rays");
        let raygen_shader_binding_table_entry = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(
                self.shadow_resource.shadow_shader_binding_tables[image_index as usize]
                    .raygen
                    .device_address(),
            )
            .size(base_alignment as u64)
            .stride(base_alignment as u64);

        let miss_shader_binding_table_entry = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(
                self.shadow_resource.shadow_shader_binding_tables[image_index as usize]
                    .miss
                    .device_address(),
            )
            .size(base_alignment as u64)
            .stride(handle_size_aligned as u64);

        let hit_shader_binding_table_entry = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(
                self.shadow_resource.shadow_shader_binding_tables[image_index as usize]
                    .hit
                    .device_address(),
            )
            .size(base_alignment as u64)
            .stride(handle_size_aligned as u64);

        let callable_shader_binding_table = vk::StridedDeviceAddressRegionKHR::default();
        unsafe {
            let extent = self
                .shadow_resource
                .shadow_render_targets
                .get(&ShadowRenderTarget::Output)
                .unwrap()
                .extent();
            self.additional_functions
                .raytracing_pipeline
                .cmd_trace_rays(
                    command_buffer,
                    &raygen_shader_binding_table_entry,
                    &miss_shader_binding_table_entry,
                    &hit_shader_binding_table_entry,
                    &callable_shader_binding_table,
                    extent.width,
                    extent.height,
                    1,
                );
        }

        log::debug!("draw deferred pass");
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines[image_index as usize].vk_pipelines()[0].clone(),
            );
        }

        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline_layouts[image_index as usize].vk_pipeline_layout(),
                0,
                &self.compute_descriptor_sets[image_index as usize].vk_descriptor_sets(),
                &[],
            );
        }

        let div_up = |a, b| (a + b - 1) / b;
        let group_count_x = div_up(extent.width, 16);
        let group_count_y = div_up(extent.height, 16);

        unsafe {
            device.cmd_dispatch(command_buffer, group_count_x, group_count_y, 1);
        }

        {
            log::debug!("depth: read -> write");
            let barriers = [
                // depth: read -> write
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(PipelineStageFlags2::ALL_GRAPHICS)
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
                    .dst_stage_mask(PipelineStageFlags2::ALL_GRAPHICS)
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .image(
                        self.graphics_render_targets[&GraphicsRenderTarget::DepthStencil].image(),
                    )
                    .subresource_range(ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            ];
            let dependency_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
            unsafe {
                device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
            }
        }

        Ok(())
    }

    fn output_render_target(&self) -> &ImageBuffer {
        &self.graphics_render_targets[&GraphicsRenderTarget::Output]
    }
}
