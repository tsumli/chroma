use crate::image::{
    Image,
    Pixels,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeFace {
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
}

fn compute_cubemap_to_equirectangular_coord(
    x: u32,
    y: u32,
    face: CubeFace,
    face_size: u32,
) -> (f32, f32) {
    let nx = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0; // Normalize to [-1, 1]
    let ny = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;

    let (vx, vy, vz) = match face {
        CubeFace::Front => (-nx, -ny, 1.0),
        CubeFace::Back => (nx, -ny, -1.0),
        CubeFace::Left => (1.0, -ny, nx),
        CubeFace::Right => (-1.0, -ny, -nx),
        CubeFace::Top => (nx, 1.0, -ny),
        CubeFace::Bottom => (-nx, -1.0, ny),
    };

    // Normalize the direction vector
    let length = (vx * vx + vy * vy + vz * vz).sqrt();
    let vx = vx / length;
    let vy = vy / length;
    let vz = vz / length;

    // Convert to spherical coordinates
    let theta = vz.atan2(vx); // Longitude (θ) [-π, π]
    let phi = vy.asin(); // Latitude (φ) [-π/2, π/2]

    // Map spherical coordinates to equirectangular image coordinates
    let u = 0.5 + theta / (2.0 * std::f32::consts::PI); // Map θ to [0, 1]
    let v = 0.5 - phi / std::f32::consts::PI; // Map φ to [0, 1]

    (u, v)
}

pub fn convert_equirectangular_to_cubemap_image(image: &Image) -> HashMap<CubeFace, Image> {
    // equirectangular image has 360 degree horizontally so face size is 1/4 of the width
    let face_size = image.width / 4;

    let pixels = match &image.pixels {
        Pixels::F32(pixels) => pixels,
        _ => panic!("Only support f32 pixel format"),
    };

    let mut cubemap = HashMap::new();

    for &face in [
        CubeFace::Front,
        CubeFace::Back,
        CubeFace::Left,
        CubeFace::Right,
        CubeFace::Top,
        CubeFace::Bottom,
    ]
    .iter()
    {
        let mut face_image = vec![0.0f32; (face_size * face_size * 4) as usize];

        for y in 0..face_size {
            for x in 0..face_size {
                // Convert cubemap face (x, y) to equirectangular (u, v)
                let (u, v) = compute_cubemap_to_equirectangular_coord(x, y, face, face_size);

                // Map equirectangular (u, v) to pixel coordinates
                let mut px =
                    (u * (image.width as f32)).clamp(0.0, (image.width - 1) as f32) as usize;
                let py = (v * (image.height as f32)).clamp(0.0, (image.height - 1) as f32) as usize;
                let mut pixel_index = (y * face_size + x) as usize * 4;

                // Handle the top/bottom faces
                match face {
                    CubeFace::Top => {
                        px = (image.width as f32 - 1.0 - px as f32) as usize;
                        pixel_index = (x * face_size + y) as usize * 4;
                    }
                    CubeFace::Bottom => {
                        pixel_index = (x * face_size + y) as usize * 4;
                    }
                    _ => {}
                }

                face_image[pixel_index] = pixels[py * image.width as usize * 4 + px * 4];
                face_image[pixel_index + 1] = pixels[py * image.width as usize * 4 + px * 4 + 1];
                face_image[pixel_index + 2] = pixels[py * image.width as usize * 4 + px * 4 + 2];
                face_image[pixel_index + 3] = pixels[py * image.width as usize * 4 + px * 4 + 3];
            }
        }

        cubemap.insert(
            face,
            Image {
                pixels: Pixels::F32(face_image),
                format: image.format,
                width: face_size,
                height: face_size,
            },
        );
    }

    cubemap
}

#[cfg(test)]
mod tests {
    use super::*;
    use chroma_base::path::get_project_root;

    #[test]
    fn test_convert_equirectangular_to_cubemap_image() {
        let proj_root = get_project_root().unwrap();
        let image = Image::from_exr(&proj_root.join("asset/hdri/golden_bay.exr"));
        let cubemap = convert_equirectangular_to_cubemap_image(&image);

        // save
        for (face, image) in cubemap.iter() {
            image.save(&proj_root.join(format!("golden_bay_{:?}.png", face)));
        }
    }
}
