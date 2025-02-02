use crate::{
    image::Image,
    light::Light,
    material::Volume,
    model::{
        ModelDescriptor,
        ModelTrait,
    },
    transform,
};
use anyhow::Result;
use chroma_base::path::get_project_root;
use gltf::{
    buffer,
    image,
    Document,
    Node,
};
use nalgebra_glm::{
    Mat4,
    Vec3,
    Vec4,
};
use std::path::PathBuf;

/// DFS traversal of the glTF scene graph to get all the nodes
fn read_object_nodes_recursive<'a>(
    node: Node<'a>,
    cur_transform: Mat4,
    nodes: &mut Vec<Node<'a>>,
    transforms: &mut Vec<Mat4>,
) {
    if node.mesh().is_some() {
        nodes.push(node.clone());
        transforms.push(cur_transform);
    }
    for child in node.children() {
        read_object_nodes_recursive(
            child.clone(),
            cur_transform * Mat4::from(node.transform().matrix()),
            nodes,
            transforms,
        );
    }
}

/// Get all the nodes in the glTF document without light nodes
fn read_object_nodes(document: &Document) -> (Vec<Node>, Vec<Mat4>) {
    let mut nodes = Vec::new();
    let mut transforms = Vec::new();
    for scenes in document.scenes() {
        for node in scenes.nodes() {
            read_object_nodes_recursive(node, Mat4::identity(), &mut nodes, &mut transforms);
        }
    }
    (nodes, transforms)
}

/// Get all the light nodes in the glTF document
fn read_light_nodes(document: &Document) -> Vec<Node> {
    let mut nodes = Vec::new();
    for scenes in document.scenes() {
        for node in scenes.nodes() {
            if node.light().is_some() {
                nodes.push(node.clone());
            }
        }
    }
    nodes
}

#[derive(Debug, Clone)]
pub struct GltfAdapter {
    document: Document,
    buffers: Vec<buffer::Data>,
    images: Vec<image::Data>,
    model_descriptor: ModelDescriptor,
    path: PathBuf,
}

impl ModelTrait for GltfAdapter {
    fn from_model_descriptor(desc: &ModelDescriptor) -> Result<Self> {
        let path = get_project_root()?.join(&desc.path);
        log::info!("Loading glTF from: {:?}", path);
        let (document, buffers, images) = gltf::import(&path)?;
        Ok(Self {
            document,
            buffers,
            images,
            model_descriptor: desc.clone(),
            path,
        })
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    fn read_positions(&self) -> Vec<Vec<[f32; 3]>> {
        let scale = &self.model_descriptor.transform.scale;
        let translation = &self.model_descriptor.transform.translation;
        let rotation = &self.model_descriptor.transform.rotation;
        let rotation = nalgebra_glm::quat_angle_axis(rotation[0], &Vec3::new(1.0, 0.0, 0.0))
            * nalgebra_glm::quat_angle_axis(rotation[1], &Vec3::new(0.0, 1.0, 0.0))
            * nalgebra_glm::quat_angle_axis(rotation[2], &Vec3::new(0.0, 0.0, 1.0));

        let mut positions_vec = Vec::new();

        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();

            for primitive in mesh.primitives() {
                let positions_iter = primitive
                    .reader(|buffer| Some(&self.buffers[buffer.index()]))
                    .read_positions()
                    .unwrap();

                let mut positions = Vec::new();
                for pos in positions_iter {
                    let position = {
                        let mut pos = Vec4::new(pos[0], pos[1], pos[2], 1.0);
                        pos = transform * pos;
                        pos = pos + Vec4::new(translation[0], translation[1], translation[2], 0.0);
                        pos = pos * scale.clone();
                        pos = nalgebra_glm::quat_rotate_vec(&rotation, &pos);
                        [pos.x, pos.y, pos.z]
                    };
                    positions.push(position);
                }

                positions_vec.push(positions);
            }
        }
        positions_vec
    }

    fn read_indices(&self) -> Vec<Vec<u32>> {
        let mut indices_vec = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let indices_iter = primitive
                    .reader(|buffer| Some(&self.buffers[buffer.index()]))
                    .read_indices()
                    .unwrap();

                let mut indices = Vec::new();
                for index in indices_iter.into_u32() {
                    indices.push(index);
                }
                indices_vec.push(indices);
            }
        }
        indices_vec
    }

    fn read_uvs(&self) -> Vec<Option<Vec<[f32; 2]>>> {
        let mut uv_vec = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();

            for primitive in mesh.primitives() {
                let uvs_iter = primitive
                    .reader(|buffer| Some(&self.buffers[buffer.index()]))
                    .read_tex_coords(0);

                if uvs_iter.is_none() {
                    uv_vec.push(None);
                    continue;
                }
                let uvs_iter = uvs_iter.unwrap();

                let mut uvs = Vec::new();
                for uv in uvs_iter.into_f32() {
                    uvs.push([uv[0], uv[1]]);
                }
                uv_vec.push(Some(uvs));
            }
        }
        uv_vec
    }

    fn read_normals(&self) -> Vec<Vec<[f32; 3]>> {
        let rotation = &self.model_descriptor.transform.rotation;
        let rotation = nalgebra_glm::quat_angle_axis(rotation[0], &Vec3::new(1.0, 0.0, 0.0))
            * nalgebra_glm::quat_angle_axis(rotation[1], &Vec3::new(0.0, 1.0, 0.0))
            * nalgebra_glm::quat_angle_axis(rotation[2], &Vec3::new(0.0, 0.0, 1.0));

        let mut normals_vec = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let normals_iter = primitive
                    .reader(|buffer| Some(&self.buffers[buffer.index()]))
                    .read_normals()
                    .unwrap();

                let mut normals = Vec::new();
                for normal in normals_iter {
                    let mut normal = Vec4::new(normal[0], normal[1], normal[2], 1.0);
                    normal = transform * normal;
                    normal = nalgebra_glm::quat_rotate_vec(&rotation, &normal);
                    normals.push([normal[0], normal[1], normal[2]]);
                }
                normals_vec.push(normals);
            }
        }
        normals_vec
    }

    fn read_tangents(&self) -> Vec<Option<Vec<[f32; 4]>>> {
        let mut tangents_vec = Vec::new();

        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let tangents_iter = primitive
                    .reader(|buffer| Some(&self.buffers[buffer.index()]))
                    .read_tangents();

                if let Some(tangent_iter) = tangents_iter {
                    let mut tangents = Vec::new();
                    for tangent in tangent_iter {
                        tangents.push([tangent[0], tangent[1], tangent[2], tangent[3]]);
                    }
                    tangents_vec.push(Some(tangents));
                } else {
                    tangents_vec.push(None);
                }
            }
        }

        tangents_vec
    }

    fn read_colors(&self) -> Vec<Option<Vec<[f32; 4]>>> {
        let mut colors_vec = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let colors_iter = primitive
                    .reader(|buffer| Some(&self.buffers[buffer.index()]))
                    .read_colors(0);

                if colors_iter.is_none() {
                    colors_vec.push(None);
                    continue;
                }
                let colors_iter = colors_iter.unwrap();

                let colors = match colors_iter {
                    gltf::mesh::util::ReadColors::RgbF32(iter) => iter
                        .map(|color| [color[0], color[1], color[2], 1.0])
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbaF32(iter) => iter
                        .map(|color| [color[0], color[1], color[2], color[3]])
                        .collect(),
                    _ => vec![[1.0, 1.0, 1.0, 1.0]],
                };
                colors_vec.push(Some(colors));
            }
        }
        colors_vec
    }

    fn read_base_colors(&self) -> (Vec<Option<Image>>, Vec<[f32; 4]>) {
        let mut sources = Vec::new();
        let mut factors = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let pbr_metallic_roughness = material.pbr_metallic_roughness();
                factors.push([
                    pbr_metallic_roughness.base_color_factor()[0],
                    pbr_metallic_roughness.base_color_factor()[1],
                    pbr_metallic_roughness.base_color_factor()[2],
                    pbr_metallic_roughness.base_color_factor()[3],
                ]);

                let base_color_texture = pbr_metallic_roughness.base_color_texture();
                if base_color_texture.is_none() {
                    sources.push(None);
                    continue;
                } else {
                    let source = base_color_texture.unwrap().texture().source();
                    sources.push(Some(source.index()));
                }
            }
        }

        let textures = sources
            .iter()
            .map(|source| {
                source.map_or(None, |source| {
                    Some(Image::from_gltf_data(self.images[source].clone()))
                })
            })
            .collect::<Vec<Option<Image>>>();

        (textures, factors)
    }

    fn read_normal_images(&self) -> Vec<Option<Image>> {
        let mut sources = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let normal_texture = material.normal_texture();
                if normal_texture.is_none() {
                    sources.push(None);
                    continue;
                } else {
                    let source = normal_texture.unwrap().texture().source();
                    sources.push(Some(source.index()));
                }
            }
        }

        let textures = sources
            .iter()
            .map(|source| {
                source.map_or(None, |source| {
                    let image = Image::from_gltf_data(self.images[source].clone());

                    // invert y-axis
                    // image.flip_y().unwrap();
                    Some(image)
                })
            })
            .collect::<Vec<_>>();

        textures
    }

    fn read_emissive_images(&self) -> Vec<Image> {
        let mut sources = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let emissive_texture = material.emissive_texture().unwrap();
                let source = emissive_texture.texture().source();
                sources.push(source.index());
            }
        }

        let textures = sources
            .iter()
            .map(|source| Image::from_gltf_data(self.images[*source].clone()))
            .collect::<Vec<Image>>();

        textures
    }

    fn read_occlusion_images(&self) -> Vec<Image> {
        let mut sources = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let occlusion_texture = material.occlusion_texture().unwrap();
                let source = occlusion_texture.texture().source();
                sources.push(source.index());
            }
        }

        let textures = sources
            .iter()
            .map(|source| Image::from_gltf_data(self.images[*source].clone()))
            .collect::<Vec<Image>>();

        textures
    }

    fn read_metallic_roughnesses(&self) -> (Vec<Option<Image>>, Vec<f32>, Vec<f32>) {
        let mut sources = Vec::new();
        let mut metallic_factors = Vec::new();
        let mut roughness_factors = Vec::new();

        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let pbr_metallic_roughness = material.pbr_metallic_roughness();
                let metallic_roughness_texture =
                    pbr_metallic_roughness.metallic_roughness_texture();
                if metallic_roughness_texture.is_none() {
                    sources.push(None);
                    continue;
                }
                let source = metallic_roughness_texture.unwrap().texture().source();
                sources.push(Some(source.index()));
                metallic_factors.push(pbr_metallic_roughness.metallic_factor());
                roughness_factors.push(pbr_metallic_roughness.roughness_factor());
            }
        }

        let textures = sources
            .iter()
            .map(|source| {
                source.map_or(None, |source| {
                    Some(Image::from_gltf_data(self.images[source].clone()))
                })
            })
            .collect::<Vec<Option<Image>>>();

        (textures, metallic_factors, roughness_factors)
    }

    fn read_transmission(&self) -> (Vec<Option<Image>>, Vec<f32>) {
        let mut texture_indices = Vec::new();
        let mut factors = Vec::new();
        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let transmission = primitive.material().transmission();
                if transmission.is_none() {
                    texture_indices.push(None);
                    continue;
                }

                let transmission = transmission.unwrap();
                let transmission_texture = transmission.transmission_texture();
                if let Some(texture) = transmission_texture {
                    texture_indices.push(Some(texture.texture().source().index()));
                } else {
                    texture_indices.push(None);
                }
                let transmission_factor = transmission.transmission_factor();
                factors.push(transmission_factor);
            }
        }

        let textures = texture_indices
            .iter()
            .map(|source| {
                source.map_or(None, |source| {
                    Some(Image::from_gltf_data(self.images[source].clone()))
                })
            })
            .collect::<Vec<Option<Image>>>();

        (textures, factors)
    }

    fn read_volume(&self) -> Vec<Option<Volume>> {
        let mut volumes = Vec::new();

        let (nodes, transforms) = read_object_nodes(&self.document);
        for (node, _transform) in nodes.iter().zip(transforms.iter()) {
            let mesh = node.mesh().unwrap();
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let volume = material.volume();
                if volume.is_none() {
                    volumes.push(None);
                    continue;
                }
                let volume = volume.unwrap();
                volumes.push(Some(Volume {
                    thickness_factor: volume.thickness_factor(),
                    attenuation_distance: volume.attenuation_distance(),
                    attenuation_color: volume.attenuation_color(),
                }));
            }
        }
        volumes
    }

    fn read_punctual_lights(&self) -> Vec<Option<Light>> {
        let mut lights = Vec::new();
        for node in read_light_nodes(&self.document) {
            assert!(node.light().is_some());
            let light = match node.light().as_ref().unwrap().kind() {
                gltf::khr_lights_punctual::Kind::Point => {
                    let light = node.light().unwrap();
                    let (position, _, _) = node.transform().decomposed();
                    let point_light = crate::light::PointLight {
                        position,
                        color: light.color(),
                        intensity: light.intensity(),
                        range: light.range().unwrap_or_default(),
                    };
                    Light::Point(point_light)
                }
                _ => {
                    log::warn!(
                        "Unsupported light type found in glTF. Only point lights are supported."
                    );
                    continue;
                }
            };
            lights.push(Some(light));
        }
        lights
    }
}
