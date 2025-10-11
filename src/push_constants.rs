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
}

pub struct PushConstantsInput<'a> {
    pub cam_pixel_to_ray: Matrix4<f64>,
    pub voxel: &'a VoxelManager,
    pub render_mode: u32,
    pub light_dir: [f32; 3], // xyz normalized; w ignored
}

pub fn build_push_constants(input: PushConstantsInput) -> PushConstants {
    // Derive scene AABB from grid origins + dims (x extent only currently; y,z start at 0)
    let mut max_x = 0.0f64;
    let mut max_y = 0.0f64;
    let mut max_z = 0.0f64;
    for (i, (dx, dy, dz)) in input.voxel.grid_dims.iter().enumerate() {
        // origin_x stored in grid infos built on GPU side; we recompute here with same logic:
        // replicate packing logic: accumulate dim_x * 1.25 up to index i
        let mut origin_x = 0.0f64;
        let mut cursor = 0.0f64;
        for j in 0..i {
            let (pdx, _pdy, _pdz) = input.voxel.grid_dims[j];
            cursor += pdx as f64 * 1.25;
        }
        origin_x = cursor;
        max_x = max_x.max(origin_x + *dx as f64);
        max_y = max_y.max(*dy as f64);
        max_z = max_z.max(*dz as f64);
    }
    if max_x <= 0.0 || max_y <= 0.0 || max_z <= 0.0 {
        let size = input.voxel.voxel_resolution as f64;
        max_x = size;
        max_y = size;
        max_z = size;
    }
    let size_vec = Vector4::new(max_x, max_y, max_z, 1.0);
    let mut scale_and_center = Matrix4::from_diagonal(&size_vec);
    scale_and_center.set_column(3, &Vector3::new(0.5 * max_x, 0.5 * max_y, 0.5 * max_z).push(1.0));
    let pixel_to_ray = scale_and_center * input.cam_pixel_to_ray;

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
    }
}
