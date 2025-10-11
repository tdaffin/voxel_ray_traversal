use bytemuck::NoUninit;
// Removed terminal progress dependency (crossterm) since we use callback-based progress.
use nalgebra::{Vector2, Vector3, Vector4};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

type Vec2 = Vector2<f32>;
type Vec3 = Vector3<f32>;
type Vec4 = Vector4<f32>;

type UVec3 = Vector3<usize>;

pub struct VoxelProgressCallbacks<'a> {
    pub cancelled: &'a AtomicBool,
    pub progress: Option<Box<dyn Fn(usize, usize) + Send + 'a>>, // (done,total)
}

pub fn ply_to_voxels_with_progress(
    path: impl AsRef<Path>, resolution: u32, cb: VoxelProgressCallbacks,
) -> Option<(Vec<u128>, Vec<u8>, (u32, u32, u32), (u32, u32, u32))> {
    let mut mesh = parse_ply(path);
    if cb.cancelled.load(Ordering::Relaxed) {
        return None;
    }
    transform_vertices(&mut mesh.vertices, resolution);
    if cb.cancelled.load(Ordering::Relaxed) {
        return None;
    }
    let (voxels, color_indices, logical_dims, storage_dims) =
        voxelize_mesh_progress(&mesh, resolution, &cb);
    if cb.cancelled.load(Ordering::Relaxed) {
        return None;
    }
    Some((voxels, color_indices, logical_dims, storage_dims))
}

pub fn ply_to_voxels(
    path: impl AsRef<Path>, resolution: u32,
) -> (Vec<u128>, Vec<u8>, (u32, u32, u32), (u32, u32, u32)) {
    ply_to_voxels_with_progress(
        path,
        resolution,
        VoxelProgressCallbacks { cancelled: &AtomicBool::new(false), progress: None },
    )
    .unwrap_or_default()
}

fn parse_ply(path: impl AsRef<Path>) -> Mesh {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);

    #[derive(Clone, Copy, NoUninit)]
    #[repr(C)]
    struct Vertex {
        inner: [f32; 3],
    }

    #[derive(Clone, Copy, NoUninit)]
    #[repr(C)]
    struct Triangle {
        inner: [i32; 3],
    }

    impl PropertyAccess for Vertex {
        fn new() -> Self {
            Vertex { inner: [0.0; 3] }
        }

        fn set_property(&mut self, key: String, property: Property) {
            // Flips the x axis and swaps y and z axes to match our coordinate system
            match (key.as_ref(), property) {
                ("x", Property::Float(v)) => self.inner[0] = -v,
                ("y", Property::Float(v)) => self.inner[2] = v,
                ("z", Property::Float(v)) => self.inner[1] = v,
                _ => (),
            }
        }
    }

    impl PropertyAccess for Triangle {
        fn new() -> Self {
            Triangle { inner: [0; 3] }
        }

        fn set_property(&mut self, key: String, property: Property) {
            if let ("vertex_indices", Property::ListInt(vec)) = (key.as_ref(), property) {
                self.inner.copy_from_slice(&vec)
            }
        }
    }

    let vertex_parser = Parser::<Vertex>::new();
    let face_parser = Parser::<Triangle>::new();

    let header = vertex_parser.read_header(&mut reader).unwrap();

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for (_ignore_key, element) in &header.elements {
        match element.name.as_ref() {
            "vertex" => {
                vertices =
                    vertex_parser.read_payload_for_element(&mut reader, element, &header).unwrap();
            }
            "face" => {
                triangles =
                    face_parser.read_payload_for_element(&mut reader, element, &header).unwrap();
            }
            _ => panic!(),
        }
    }

    let vertices: Vec<Vec3> = bytemuck::cast_vec(vertices);
    let triangles: Vec<[i32; 3]> = bytemuck::cast_vec(triangles);

    Mesh { vertices, triangles }
}

fn transform_vertices(vertices: &mut [Vec3], resolution: u32) {
    let mut min = Vec3::from_element(f32::MAX);
    let mut max = Vec3::from_element(f32::MIN);

    for vertex in vertices.iter() {
        for i in 0..3 {
            max[i] = max[i].max(vertex[i]);
            min[i] = min[i].min(vertex[i]);
        }
    }

    let range = max - min;
    let size = (range.x).max(range.y).max(range.z);

    for vertex in vertices.iter_mut() {
        for i in 0..3 {
            vertex[i] = ((vertex[i] - min[i]) / size) * resolution as f32;
            vertex[i] = vertex[i].clamp(0.0, (resolution as f32).next_down());
        }
    }
}

// Each u128 is an rgba32ui on the GPU in a 3D texture.
// Each texel is 4x4x8 voxels, and each channel is 1x4x8 voxels.

fn voxelize_mesh_progress(
    mesh: &Mesh, resolution: u32, cb: &VoxelProgressCallbacks,
) -> (Vec<u128>, Vec<u8>, (u32, u32, u32), (u32, u32, u32)) {
    // For non-cubic storage we preserve logical dims (resolution^3 target still uniform for ply)
    // but future per-axis scaling could set these independently. Keep interface flexible.
    let dx = resolution;
    let dy = resolution;
    let dz = resolution;
    // Packed texel extents (4x4x8 voxels per texel)
    let storage_w = (dx + 3) / 4; // ceil div
    let storage_h = (dy + 3) / 4;
    let storage_d = (dz + 7) / 8;
    let packed_texel_count = (storage_w * storage_h * storage_d) as usize;
    let mut voxels = vec![0u128; packed_texel_count];
    let mut color_indices = vec![0u8; (dx * dy * dz) as usize];
    let total = mesh.triangles.len();
    let mut processed = 0usize;
    for triangle in &mesh.triangles {
        if cb.cancelled.load(Ordering::Relaxed) {
            break;
        }
        let a = &mesh.vertices[triangle[0] as usize];
        let b = &mesh.vertices[triangle[1] as usize];
        let c = &mesh.vertices[triangle[2] as usize];
        let helper = Helper::new(a, b, c);
        helper.visit_intersecting_voxels(|x, y, z| {
            if x >= dx as usize || y >= dy as usize || z >= dz as usize {
                return;
            }
            // Flatten into packed texel index using storage extents
            let tx = x / 4;
            let ty = y / 4;
            let tz = z / 8;
            let texel = (tx + (ty + tz * storage_h as usize) * storage_w as usize) as usize;
            let bit = (x % 4) * 32 + (y % 4) + (z % 8) * 4;
            voxels[texel] |= 1u128 << bit;
            // Procedural local palette index: gradient based on z low 4 bits (0..15)
            let idx_local = (z as u8) & 0x0F;
            color_indices[(z * dy as usize + y) * dx as usize + x] = idx_local;
        });
        processed += 1;
        if processed % 256 == 0 {
            // throttle callbacks
            if let Some(p) = &cb.progress {
                p(processed, total);
            }
        }
    }
    if let Some(p) = &cb.progress {
        p(processed, total);
    }
    (voxels, color_indices, (dx, dy, dz), (storage_w, storage_h, storage_d))
}

struct Helper {
    // Bounds
    min: Vec3,
    max: Vec3,

    // Tests
    n: Vec3,
    lower: f32,
    upper: f32,
    tests: [Vec4; 9],
}

impl Helper {
    fn new(a: &Vec3, b: &Vec3, c: &Vec3) -> Self {
        let n = (b - a).cross(&(c - a));
        let signum = n.map(f32::signum);

        let min = a.zip_zip_map(b, c, |a, b, c| a.min(b).min(c));
        let max = a.zip_zip_map(b, c, |a, b, c| a.max(b).max(c));

        let nd1 = n.sum();
        let nda = n.dot(a);
        let nds = n.dot(&signum);

        let lower = nda - (nd1 + nds) * 0.5;
        let upper = nda - (nd1 - nds) * 0.5;

        let mut tests = [Vec4::zeros(); 9];

        let tri = [a, b, c];
        for i in 0..3 {
            let pos = tri[i];
            let edge = tri[(i + 1) % 3] - tri[i];
            for a in 0..3 {
                let b = (a + 1) % 3;
                let c = (a + 2) % 3;

                let n = Vec2::new(-edge[b], edge[a]) * signum[c];
                let p = Vec2::new(pos[a], pos[b]);
                let d = n.dot(&p) - n.x.max(0.0) - n.y.max(0.0);

                tests[a * 3 + i][a] = n.x;
                tests[a * 3 + i][b] = n.y;
                tests[a * 3 + i].w = d;
            }
        }

        Self { min, max, n, lower, upper, tests }
    }

    fn visit_intersecting_voxels<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize, usize),
    {
        let min = self.min.map(|x| x as usize);
        let max = self.max.map(|x| x as usize);

        for z in min.z..=max.z {
            let mut y_started = false;
            for y in min.y..=max.y {
                let mut x_started = false;
                for x in min.x..=max.x {
                    let coord = UVec3::new(x, y, z);
                    let intersects = self.intersect(coord);
                    if intersects {
                        f(x, y, z);
                    }

                    if x_started && !intersects {
                        break;
                    }
                    x_started = intersects;
                }
                if y_started && !x_started {
                    break;
                }
                y_started = x_started;
            }
        }
    }

    fn intersect(&self, p: UVec3) -> bool {
        let p = p.cast::<f32>();

        let d = self.n.dot(&p);
        if d < self.lower || d > self.upper {
            return false;
        }

        for test in &self.tests {
            if test.xyz().dot(&p) < test.w {
                return false;
            }
        }

        true
    }
}

struct Mesh {
    vertices: Vec<Vec3>,
    triangles: Vec<[i32; 3]>,
}

// Removed unused legacy terminal progress machinery and blocking voxelizer.
