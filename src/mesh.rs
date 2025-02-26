use ocl_macros::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{file::ByteStream, lbm::Lbm};
use std::{f32::consts::PI, ops::{Add, AddAssign, Mul, Sub}};

pub enum ModelType {
    Solid,
    Magnet,
    Charged,
}

#[derive(Debug, Clone, Copy)]
/// 3D Vector type
struct F32_3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Default for F32_3 {
    fn default() -> Self {
        F32_3 { x: 0.0, y: 0.0, z: 0.0 }
    }
}

#[derive(Debug, Clone, Copy)]
/// 3D Matrix type
struct F32_3_3 {
    xx: f32, yx: f32, zx: f32,
    xy: f32, yy: f32, zy: f32,
    xz: f32, yz: f32, zz: f32,
}

impl Default for F32_3_3 {
    fn default() -> Self {
        F32_3_3 { xx: 1.0, yx: 1.0, zx: 1.0, xy: 1.0, yy: 1.0, zy: 1.0, xz: 1.0, yz: 1.0, zz: 1.0 }
    }
}

impl F32_3_3 {
    fn construct_rotation_matrix(rx: f32, ry: f32, rz: f32) -> Self {
        let al = rz / 180.0 * 2.0 * PI;
        let be = ry / 180.0 * 2.0 * PI; 
        let ga = rx / 180.0 * 2.0 * PI; 
        F32_3_3 {
            xx: be.cos()*ga.cos(), yx: al.sin()*be.sin()*ga.cos()-al.cos()*ga.sin(), zx: al.cos()*be.sin()*ga.cos()+al.sin()*ga.sin(),
            xy: be.cos()*ga.sin(), yy: al.sin()*be.sin()*ga.sin()+al.cos()*ga.cos(), zy: al.cos()*be.sin()*ga.sin()-al.sin()*ga.cos(),
            xz: -be.sin(),         yz: al.sin()*be.cos(),                            zz: al.cos()*be.cos() }
    }
}

pub struct Mesh {
    triangle_number: u32,
    center: F32_3,
    p_min:  F32_3,
    p_max:  F32_3,
    p0: Vec<F32_3>,
    p1: Vec<F32_3>,
    p2: Vec<F32_3>,
}

#[allow(unused)]
impl Mesh {
    fn new(triangle_number: u32, center: F32_3) -> Mesh {
        Mesh {
            triangle_number,
            center,
            p_min: center,
            p_max: center,
            p0: vec![F32_3::default(); triangle_number as usize],
            p1: vec![F32_3::default(); triangle_number as usize],
            p2: vec![F32_3::default(); triangle_number as usize],
        }
    }

    /// Update the mesh bounding box
    fn update_bounds(&mut self) {
        self.p_min = self.p0[0];
        self.p_max = self.p0[0];

        for i in 0..self.triangle_number as usize {
            let p0i = self.p0[i];
            let p1i = self.p1[i];
            let p2i = self.p2[i];
            self.p_min.x = self.p_min.x.min(p2i.x.min(p1i.x.min(p0i.x)));
            self.p_min.y = self.p_min.y.min(p2i.y.min(p1i.y.min(p0i.y)));
            self.p_min.z = self.p_min.z.min(p2i.z.min(p1i.z.min(p0i.z)));
            self.p_max.x = self.p_max.x.max(p2i.x.max(p1i.x.max(p0i.x)));
            self.p_max.y = self.p_max.y.max(p2i.y.max(p1i.y.max(p0i.y)));
            self.p_max.z = self.p_max.z.max(p2i.z.max(p1i.z.max(p0i.z)));
        }
    }

    fn scale(&mut self, scale: f32) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] = scale*(self.p0[i]-self.center)+self.center;
			self.p1[i] = scale*(self.p1[i]-self.center)+self.center;
			self.p2[i] = scale*(self.p2[i]-self.center)+self.center;
		}
		self.p_min = scale*(self.p_min-self.center)+self.center;
		self.p_max = scale*(self.p_max-self.center)+self.center;
    }

    fn translate(&mut self, translation: F32_3) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] += translation;
			self.p1[i] += translation;
			self.p2[i] += translation;
		}
		self.center += translation;
		self.p_min += translation;
		self.p_max += translation;
    }

    fn rotate(&mut self, rotation: F32_3_3) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] = rotation*(self.p0[i]-self.center)+self.center;
			self.p1[i] = rotation*(self.p1[i]-self.center)+self.center;
			self.p2[i] = rotation*(self.p2[i]-self.center)+self.center;
		}
		self.update_bounds();
    }

    fn set_center(&mut self, center: F32_3) {
        self.center = center;
    }

    fn get_center(&self) -> F32_3 {
        return self.center;
    }

    fn get_bounding_box_size(&self) -> F32_3 {
        return self.p_max - self.p_min;
    }

    fn get_bounding_box_center(&self) -> F32_3 {
        return 0.5 *(self.p_min + self.p_max);
    }

    fn get_min_size(&self) -> f32 {
        (self.p_max.x-self.p_min.x).min((self.p_max.y-self.p_min.y).min(self.p_max.z-self.p_min.z))
    }

    fn get_max_size(&self) -> f32 {
        (self.p_max.x-self.p_min.x).max((self.p_max.y-self.p_min.y).max(self.p_max.z-self.p_min.z))
    }

    /// Get scale factor so that model fits into specified box size
    fn get_scale_for_box_fit(&self, box_size: F32_3) -> f32 {
        (box_size.x/(self.p_max.x-self.p_min.x)).min( (box_size.y/(self.p_max.y-self.p_min.y)).min(box_size.z/(self.p_max.z-self.p_min.z)) )
    }

    /// Returns the direction on which the bounding box is smallest
    fn get_smallest_bb_direction() {

    }

    fn read_stl_raw(path: &str, reposition: bool, box_size: F32_3, center: F32_3, rotation: F32_3_3, size: f32) -> Mesh {
        let mut stream = ByteStream::from_buffer(&std::fs::read(path).expect(&format!("Error: could not open or find file \"{}\"", path)));
        stream.skip_bytes(80);
        let triangle_number = stream.next_u32();
        if triangle_number > 0 && stream.length() as u32 == 84+50*triangle_number {
            println!("Loading STL file \"{}\" with {} triangles", path, triangle_number);
        } else {
            println!("Error: File \"{}\" is corrupted or unsupported. Only binary .stl files are supported. Aborting.", path);
            panic!("Mesh import failed");
        }

        let mut mesh = Mesh::new(triangle_number, center);

        for i in 0..triangle_number as usize { // Iterate over triangles
            stream.skip_bytes(12); // Skip normal vector
            mesh.p0[i] = rotation*F32_3 {x: stream.next_f32(), y: stream.next_f32(), z: stream.next_f32()};
            mesh.p1[i] = rotation*F32_3 {x: stream.next_f32(), y: stream.next_f32(), z: stream.next_f32()};
            mesh.p2[i] = rotation*F32_3 {x: stream.next_f32(), y: stream.next_f32(), z: stream.next_f32()};
            stream.skip_bytes(2); // Skip attribute bits
        }
        assert!(stream.at_end(), "File was not completely read");
        mesh.update_bounds();

        let scale;
        if size == 0.0 {
            scale = mesh.get_scale_for_box_fit(box_size);
        } else if size > 0.0 {
            scale = size/mesh.get_max_size();
        } else {
            scale = -size
        }

        let offset = if reposition { -0.5 * (mesh.p_min+mesh.p_max) } else { F32_3::default() };
        for i in 0..triangle_number as usize {
            mesh.p0[i] = center+scale*(offset+mesh.p0[i]);
            mesh.p1[i] = center+scale*(offset+mesh.p1[i]);
            mesh.p2[i] = center+scale*(offset+mesh.p2[i]);
        }
        mesh.update_bounds();

        mesh
    }
}

impl Lbm {
    /// Add an STL model as a mesh to the LBM
    /// cx, cy, cz are center coordinates
    /// rx, ry, rz are rotations in degrees
    pub fn import_mesh(&mut self, path: &str, scale: f32, cx: f32, cy: f32, cz: f32, rx: f32, ry: f32, rz: f32) {
        let box_size = F32_3 {x: 1.0, y: 1.0, z: 1.0};
        let center = F32_3 {x: cx, y: cy, z: cz};
        let rot_matrix = F32_3_3::construct_rotation_matrix(rx, ry, rz);
        println!("{:?}", rot_matrix);
        self.meshes.push(Mesh::read_stl_raw(path, false, box_size, center, rot_matrix, -scale.abs()));
    }

    /// Add an STL model as a mesh to the LBM
    /// cx, cy, cz are center coordinates
    /// rx, ry, rz are rotations in degrees
    pub fn import_mesh_reposition(&mut self, path: &str, cx: f32, cy: f32, cz: f32, rx: f32, ry: f32, rz: f32, size: f32) {
        let box_size = F32_3 {x: 1.0, y: 1.0, z: 1.0};
        let center = F32_3 {x: cx, y: cy, z: cz};
        let rot_matrix = F32_3_3::construct_rotation_matrix(rx, ry, rz);
        self.meshes.push(Mesh::read_stl_raw(path, true, box_size, center, rot_matrix, size));
    }

    #[allow(unused)]
    pub fn voxelise_mesh(&mut self, index: usize, ctype: ModelType) {
        let mesh = &self.meshes[index];

        let nx = self.config.n_x; let ny = self.config.n_y; let nz = self.config.n_z;
        let x0 = mesh.p_min.x; let x1 = mesh.p_max.x;
        let y0 = mesh.p_min.y; let y1 = mesh.p_max.y; 
        let z0 = mesh.p_min.z; let z1 = mesh.p_max.z;
        
        let direction = if (y1-y0)*(z1-z0) < (x1-x0)*(z1-z0) && (y1-y0)*(z1-z0) < (x1-x0)*(y1-y0) { 0 } // x direction has smallest area
            else if (x1-x0)*(z1-z0) < (x1-x0)*(y1-y0) { 1 } // y direction has smallest area
            else { 2 }; // z direction has smallest area
        
        let tn = mesh.triangle_number;
        
        let flag = 1u8;
        
        for d in &mut self.domains {
            let an = match direction {
                0 => d.n_y * d.n_z,
                1 => d.n_x * d.n_z,
                2 => d.n_x * d.n_y,
                _ => 0
            };

            let flags = bget!(d.flags);

            (0..an as usize).into_par_iter().for_each(|a| {
                let xyz = if direction == 0 {
                    F32_3 { x: (x0), y: (a as u32%ny) as f32, z: (a as u32/ny) as f32}
                } else if direction == 1 {
                    F32_3 { x: 0.0, y: (a as u32%ny) as f32, z: (a as u32/ny) as f32}
                } else {
                    F32_3 { x: 0.0, y: (a as u32%ny) as f32, z: (a as u32/ny) as f32}
                };
                
                deborrow(&flags)[a] = flag;
            });

            bwrite!(d.flags, flags)
        }
    }
}

// Add vector to vector
impl Add<F32_3> for F32_3{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        F32_3 {x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z}
    }
}

// Subtract vector from vector
impl Sub<F32_3> for F32_3{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        F32_3 {x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z}
    }
}

// Scale vector by scalar
impl Mul<f32> for F32_3{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        F32_3 {x: self.x * rhs, y: self.y * rhs, z: self.z * rhs}
    }
}

// Scale vector by scalar
impl Mul<F32_3> for f32{
    type Output = F32_3;

    fn mul(self, rhs: F32_3) -> Self::Output {
        F32_3 {x: self * rhs.x, y: self * rhs.y, z: self * rhs.z}
    }
}

// Add and assign vector to vector
impl AddAssign<F32_3> for F32_3{

    fn add_assign(&mut self, rhs: F32_3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

// Multiply matrix with vector
impl Mul<F32_3> for F32_3_3{
    type Output = F32_3;

    fn mul(self, v: F32_3) -> Self::Output {
        F32_3 {
            x: self.xx*v.x + self.xy*v.y + self.xz*v.z,
            y: self.yx*v.x + self.yy*v.y + self.yz*v.z,
            z: self.zx*v.x + self.zy*v.y + self.zz*v.z,
        }
    }
}

// Multiply vector with matrix
impl Mul<F32_3_3> for F32_3{
    type Output = F32_3;

    fn mul(self, m: F32_3_3) -> Self::Output {
        F32_3 {
            x: self.x*m.xx + self.y*m.yx + self.z*m.zx,
            y: self.x*m.xy + self.y*m.yy + self.z*m.zy,
            z: self.x*m.xz + self.y*m.yz + self.z*m.zz,
        }
    }
}

#[inline]
fn deborrow<'b, T>(r: &T) -> &'b mut T {
    // Needed to access vector fields in parallel (3 components per index).
    // This is safe, because no indecies are accessed multiple times
    unsafe {
        #[allow(mutable_transmutes)]
        std::mem::transmute(r)
    }
}
