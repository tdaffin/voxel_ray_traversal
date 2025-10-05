use crate::model::Model;
use crate::voxel_job::VoxelManager;
use nalgebra::{Matrix4, Vector3, Vector4};

use vulkano::buffer::BufferContents;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub struct PushConstants {
    pub pixel_to_ray: Matrix4<f32>,
    pub light_dir: [f32; 4],
    pub voxel_count: u32,
    pub render_mode: u32,
    pub _pad: [u32; 2],
    pub resolutions: [u32; 3],
    pub _pad2: u32,
}

pub struct PushConstantsInput<'a> {
    pub cam_pixel_to_ray: Matrix4<f64>,
    pub voxel: &'a VoxelManager,
    pub render_mode: u32,
    pub light_dir: [f32; 3], // xyz normalized; w ignored
}

pub fn build_push_constants(input: PushConstantsInput) -> PushConstants {
    let size = input.voxel.voxel_resolution as f64;
    let mut scale_and_center = Matrix4::from_diagonal(&Vector4::from_element(size));
    scale_and_center.set_column(3, &Vector3::from_element(0.5 * size).push(1.0));
    let pixel_to_ray = scale_and_center * input.cam_pixel_to_ray;

    let voxel_count = input.voxel.active_voxel_grids.min(Model::ALL.len() as u32).max(1).min(3);
    let mut resolutions = [0u32; 3];
    for (i, r) in input.voxel.grid_resolutions.iter().take(3).enumerate() {
        resolutions[i] = *r;
    }
    if voxel_count as usize > input.voxel.grid_resolutions.len() {
        resolutions[0] = input.voxel.voxel_resolution;
    }

    let ld = {
        let v = input.light_dir;
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        [v[0] / len, v[1] / len, v[2] / len, 0.0]
    };

    PushConstants {
        pixel_to_ray: pixel_to_ray.cast(),
        light_dir: ld,
        voxel_count,
        render_mode: input.render_mode,
        _pad: [0, 0],
        resolutions,
        _pad2: 0,
    }
}
