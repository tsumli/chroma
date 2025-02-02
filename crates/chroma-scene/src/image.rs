use anyhow::{
    bail,
    ensure,
    Result,
};
use exr::prelude::*;
use std::{
    path::Path,
    u8,
};

fn normalize_pixel(pixel: f32, min: f32, max: f32) -> u8 {
    ((pixel - min) / (max - min) * 255.0) as u8
}

#[derive(Debug, Clone)]
pub enum Pixels {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct Image {
    pub pixels: Pixels,
    pub format: gltf::image::Format,
    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn add_alpha_if_not_exist(&mut self) {
        if self.format != gltf::image::Format::R8G8B8 {
            return;
        }
        let pixels_original = match &self.pixels {
            Pixels::U8(pixels) => pixels.clone(),
            _ => return,
        };

        let mut pixels = Vec::new();
        for i in 0..pixels_original.len() / 3 {
            pixels.push(pixels_original[i * 3]);
            pixels.push(pixels_original[i * 3 + 1]);
            pixels.push(pixels_original[i * 3 + 2]);
            pixels.push(255);
        }
        self.pixels = Pixels::U8(pixels);
        self.format = gltf::image::Format::R8G8B8A8;
    }

    pub fn from_exr(path: &std::path::Path) -> Self {
        assert!(path.exists());
        let exr = read_first_rgba_layer_from_file(
            path, // instantiate your image type with the size of the image in file
            |resolution, _| {
                let default_pixel = [0.0, 0.0, 0.0, 0.0];
                let empty_line = vec![default_pixel; resolution.width()];
                let empty_image = vec![empty_line; resolution.height()];
                empty_image
            },
            // transfer the colors from the file to your image type,
            // requesting all values to be converted to f32 numbers (you can also directly use f16
            // instead) and you could also use `Sample` instead of `f32` to keep the
            // original data type from the file
            |pixel_vector, position, (r, g, b, a): (f32, f32, f32, f32)| {
                pixel_vector[position.y()][position.x()] = [r, g, b, a]
            },
        )
        .unwrap();

        let width = exr.attributes.display_window.size.width() as u32;
        let height = exr.attributes.display_window.size.height() as u32;

        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for line in exr.layer_data.channel_data.pixels {
            for pixel in line {
                pixels.push(pixel[0]);
                pixels.push(pixel[1]);
                pixels.push(pixel[2]);
                pixels.push(pixel[3]);
            }
        }

        Self {
            pixels: Pixels::F32(pixels),
            format: gltf::image::Format::R32G32B32A32FLOAT,
            width,
            height,
        }
    }

    pub fn from_gltf_data(data: gltf::image::Data) -> Self {
        Self {
            pixels: Pixels::U8(data.pixels),
            format: data.format,
            width: data.width,
            height: data.height,
        }
    }

    /// Flip y axis of the image.
    /// See https://github.com/KhronosGroup/glTF-Sample-Models/blob/d7a3cc8e51d7c573771ae77a57f16b0662a905c6/2.0/NormalTangentTest/README.md?plain=1#L25
    /// for the example usage.
    pub fn flip_y(&mut self) -> Result<()> {
        let pixels = match &self.pixels {
            Pixels::U8(pixels) => pixels.clone(),
            _ => bail!("flip_y only supports u8 pixels"),
        };
        ensure!(
            self.format == gltf::image::Format::R8G8B8,
            "flip_y only supports R8G8B8 format"
        );

        let mut new_pixels = Vec::with_capacity(pixels.len());
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel_index = (y * self.width + x) as usize * 3;
                new_pixels.push(pixels[pixel_index]);
                new_pixels.push(u8::MAX - pixels[pixel_index + 1]);
                new_pixels.push(pixels[pixel_index + 2]);
            }
        }
        self.pixels = Pixels::U8(new_pixels);
        Ok(())
    }

    pub fn save(&self, path: &Path) {
        let mut img = image::RgbaImage::new(self.width, self.height);
        match &self.pixels {
            Pixels::U8(pixels) => {
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    let pixel_index = (y * self.width + x) as usize * 4;
                    let r = pixels[pixel_index];
                    let g = pixels[pixel_index + 1];
                    let b = pixels[pixel_index + 2];
                    let a = pixels[pixel_index + 3];
                    *pixel = image::Rgba([r, g, b, a]);
                }
            }
            Pixels::F32(pixels) => {
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    let pixel_index = (y * self.width + x) as usize * 4;
                    // gamma correction
                    let r = (pixels[pixel_index] as f32 / 255.0).powf(1.0 / 2.2) * 255.0;
                    let g = (pixels[pixel_index + 1] as f32 / 255.0).powf(1.0 / 2.2) * 255.0;
                    let b = (pixels[pixel_index + 2] as f32 / 255.0).powf(1.0 / 2.2) * 255.0;
                    let a = (pixels[pixel_index + 3] as f32 / 255.0).powf(1.0 / 2.2) * 255.0;
                    *pixel = image::Rgba([r as u8, g as u8, b as u8, a as u8]);
                }
            }
        };
        img.save(path).unwrap();
    }
}

pub fn create_blank_image(width: u32, height: u32) -> Image {
    let pixels = vec![u8::MAX; (width * height * 4) as usize];
    Image {
        pixels: Pixels::U8(pixels),
        format: gltf::image::Format::R8G8B8A8,
        width,
        height,
    }
}
