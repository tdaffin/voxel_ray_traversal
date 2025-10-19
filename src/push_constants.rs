use crate::voxel_job::VoxelManager;
use nalgebra::{Matrix4, Vector3};

use vulkano::buffer::BufferContents;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub struct PushConstants {
    pub pixel_to_ray: Matrix4<f32>,
    pub light_dir: [f32; 4],
    pub voxel_count: u32,
    pub render_mode: u32,
    pub always_instant: u32,
}

pub struct PushConstantsInput<'a> {
    pub cam_pixel_to_ray: Matrix4<f64>,
    pub voxel: &'a VoxelManager,
    pub render_mode: u32,
    pub light_dir: [f32; 3], // xyz normalized; w ignored
    pub always_instant: bool,
}

pub fn build_push_constants(input: PushConstantsInput) -> PushConstants {
    // Derive scene AABB from grid origins + dims (x extent only currently; y,z start at 0)
    let mut max_x = 0.0f64;
    let mut max_y = 0.0f64;
    let mut max_z = 0.0f64;
    // Reconstruct 2D packing (must match voxel_job.rs packing heuristic):
    let mut cursor = 0.0f64;
    let mut row_y = 0.0f64;
    let mut row_max_height = 0.0f64;
    for (dx, dy, dz) in input.voxel.grid_dims.iter() {
        // If adding this would exceed threshold, wrap to next row.
        if cursor > 0.0 && cursor + (*dx as f64 * 1.25) > 512.0 {
            // threshold mirrors job.rs
            // finalize previous row
            max_x = max_x.max(cursor);
            row_y += row_max_height;
            cursor = 0.0;
            row_max_height = 0.0;
        }
        let origin_x = cursor;
        let origin_y = row_y;
        max_x = max_x.max(origin_x + *dx as f64);
        max_y = max_y.max(origin_y + *dy as f64);
        max_z = max_z.max(*dz as f64);
        cursor += *dx as f64 * 1.25;
        row_max_height = row_max_height.max(*dy as f64 * 1.25);
    }
    // account last row
    max_x = max_x.max(cursor);
    if max_x <= 0.0 || max_y <= 0.0 || max_z <= 0.0 {
        let size = input.voxel.voxel_resolution as f64;
        max_x = size;
        max_y = size;
        max_z = size;
    }
    // Uniform scale to largest dimension to avoid anisotropic distortion.
    let largest = max_x.max(max_y).max(max_z).max(1.0);
    let mut scale_and_center64 = Matrix4::<f64>::identity();
    scale_and_center64[(0, 0)] = largest;
    scale_and_center64[(1, 1)] = largest;
    scale_and_center64[(2, 2)] = largest;
    scale_and_center64
        .set_column(3, &Vector3::new(0.5 * largest, 0.5 * largest, 0.5 * largest).push(1.0));
    let pixel_to_ray = scale_and_center64 * input.cam_pixel_to_ray;

    let voxel_count = input.voxel.active_voxel_grids.max(1); // no fixed cap now

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
        always_instant: input.always_instant as u32,
    }
}
