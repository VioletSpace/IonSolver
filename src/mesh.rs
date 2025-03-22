use log::{error, info};
use ocl_macros::*;

use crate::{file::ByteStream, lbm::{Lbm, LbmDomain}};
use std::{f32::consts::PI, ops::{Add, AddAssign, Mul, Sub}, time::Instant};

#[derive(Clone, Copy)]
#[allow(unused)]
pub enum ModelType {
    Solid,
    Magnet {magnetization: (f32, f32, f32)},
    Charged {charge: f32},
    ChargedECR {charge: f32},
}

#[derive(Debug, Clone, Copy)]
/// 3D Vector type
pub struct F32_3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Returns a 3D vector
pub fn v3(x: f32, y: f32, z: f32) -> F32_3 {
    F32_3 { x, y, z }
}

impl Default for F32_3 {
    fn default() -> Self {
        F32_3 { x: 0.0, y: 0.0, z: 0.0 }
    }
}

#[derive(Debug, Clone, Copy)]
/// 3D Matrix type
pub struct F32_3_3 {
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
    /// Construct a rotation matrix from rx-z in radians
    fn construct_rotation_matrix(rx: f32, ry: f32, rz: f32) -> Self {
          Self::rotm_around_v(F32_3 {x: 1.0, y: 0.0, z: 0.0}, rx)
        * Self::rotm_around_v(F32_3 {x: 0.0, y: 1.0, z: 0.0}, ry)
        * Self::rotm_around_v(F32_3 {x: 0.0, y: 0.0, z: 1.0}, rz)
    }

    fn rotm_around_v(v: F32_3, r: f32) -> Self {
        let sr = r.sin();
        let cr = r.cos();
        fn sq(x: f32) -> f32 { x * x }
        F32_3_3 {
            xx: sq(v.x)+(1.0-sq(v.x))*cr, xy: v.x*v.y*(1.0-cr)-v.z*sr,  xz: v.x*v.z*(1.0-cr)+v.y*sr,
            yx: v.x*v.y*(1.0-cr)+v.z*sr,  yy: sq(v.y)+(1.0-sq(v.y))*cr, yz: v.y*v.z*(1.0-cr)-v.x*sr,
            zx: v.x*v.z*(1.0-cr)-v.y*sr,  zy: v.y*v.z*(1.0-cr)+v.x*sr,  zz: sq(v.z)+(1.0-sq(v.z))*cr,
        }
    }
}

pub struct Mesh {
    pub triangle_number: u32,
    pub center: F32_3,
    pub p_min:  F32_3,
    pub p_max:  F32_3,
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

    pub fn scale(&mut self, scale: f32) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] = scale*(self.p0[i]-self.center)+self.center;
			self.p1[i] = scale*(self.p1[i]-self.center)+self.center;
			self.p2[i] = scale*(self.p2[i]-self.center)+self.center;
		}
		self.p_min = scale*(self.p_min-self.center)+self.center;
		self.p_max = scale*(self.p_max-self.center)+self.center;
    }

    pub fn translate(&mut self, translation: F32_3) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] += translation;
			self.p1[i] += translation;
			self.p2[i] += translation;
		}
		self.center += translation;
		self.p_min += translation;
		self.p_max += translation;
    }

    pub fn rotate(&mut self, rotation: F32_3_3) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] = rotation*(self.p0[i]-self.center)+self.center;
			self.p1[i] = rotation*(self.p1[i]-self.center)+self.center;
			self.p2[i] = rotation*(self.p2[i]-self.center)+self.center;
		}
		self.update_bounds();
    }

    pub fn set_center(&mut self, center: F32_3) {
        self.center = center;
    }

    pub fn get_center(&self) -> F32_3 {
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
            info!("Loading STL file \"{}\" with {} triangles", path, triangle_number);
        } else {
            error!("File \"{}\" is corrupted or unsupported. Only binary .stl files are supported. Aborting.", path);
            panic!("Mesh import failed");
        }

        let now = Instant::now();

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

        info!("Successfully imported \"{path}\" in {}ms", now.elapsed().as_millis());

        mesh
    }
}

impl Lbm {
    /// Add an STL model as a mesh to the LBM.
    /// 
    /// scale is the model scaling factor. It is expected that models are correctly scaled in SI units
    /// already, SI unit to simulation unit conversion is performed automatically. If you want to import
    /// a model with the inherent scaling, set scale to 1.
    /// 
    /// ox, oy, oz are the model origin coordinates in the simulation.
    /// 
    /// rx, ry, rz are rotations in degrees.
    pub fn import_mesh(&mut self, path: &str, scale: f32, ox: f32, oy: f32, oz: f32, rx: f32, ry: f32, rz: f32) {
        let box_size = F32_3 {x: 1.0, y: 1.0, z: 1.0};
        let center = F32_3 {x: ox, y: oy, z: oz};
        let rot_matrix = F32_3_3::construct_rotation_matrix(rx * PI / 180.0, ry * PI / 180.0, rz * PI / 180.0);
        let scale_lu = self.config.units.len_si_lu(scale);
        self.meshes.push(Mesh::read_stl_raw(path, false, box_size, center, rot_matrix, -scale_lu.abs()));
    }

    /// Add an STL model as a mesh to the LBM.
    /// 
    /// cx, cy, cz are the model center coordinates in the simulation.
    /// 
    /// rx, ry, rz are rotations in degrees.
    /// 
    /// size is the mesh size in simulation units on its longest axis.
    pub fn import_mesh_reposition(&mut self, path: &str, cx: f32, cy: f32, cz: f32, rx: f32, ry: f32, rz: f32, size: f32) {
        let box_size = F32_3 {x: self.config.n_x as f32, y: self.config.n_y as f32, z: self.config.n_z as f32};
        let center = F32_3 {x: cx, y: cy, z: cz};
        let rot_matrix = F32_3_3::construct_rotation_matrix(rx * PI / 180.0, ry * PI / 180.0, rz * PI / 180.0);
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

        info!("Voxelising mesh with {tn} triangles on device.");
        
        for d in &mut self.domains {
            d.voxelize_mesh_on_device(mesh, ctype);
        }
        self.finish_queues();
    }
}

impl LbmDomain {
    fn voxelize_mesh_on_device(&mut self, mesh: &Mesh, ctype: ModelType) {
        self.p0 = buffer!(&self.queue, [mesh.triangle_number * 3], 0.0f32);
        bwrite!(self.p0, mesh.p0.iter().flat_map(|p| [p.x, p.y, p.z]).collect::<Vec<f32>>());
        self.p1 = buffer!(&self.queue, [mesh.triangle_number * 3], 0.0f32);
        bwrite!(self.p1, mesh.p1.iter().flat_map(|p| [p.x, p.y, p.z]).collect::<Vec<f32>>());
        self.p2 = buffer!(&self.queue, [mesh.triangle_number * 3], 0.0f32);
        bwrite!(self.p2, mesh.p2.iter().flat_map(|p| [p.x, p.y, p.z]).collect::<Vec<f32>>());
        let x0 = mesh.p_min.x-2.0;
        let y0 = mesh.p_min.y-2.0;
        let z0 = mesh.p_min.z-2.0;
        let x1 = mesh.p_max.x+2.0;
        let y1 = mesh.p_max.y+2.0;
        let z1 = mesh.p_max.z+2.0;
        let bbu = vec![f32::from_bits(mesh.triangle_number), x0, y0, z0, x1, y1, z1];
        bwrite!(self.bbu, bbu);

        let direction: u32 = { // choose direction of minimum bounding-box cross-section area
            let c = [(y1-y0)*(z1-z0), (z1-z0)*(x1-x0), (x1-x0)*(y1-y0)];
            if c[0] < c[1] && c[0] < c[2] {
                0
            } else if c[1] < c[2] {
                1
            } else {
                2
            }
        };
        let a = [self.n_y * self.n_z, self.n_x * self.n_z, self.n_x * self.n_y];

        let flag: u8 = match ctype {
            ModelType::Solid => 0b00000001,
            ModelType::Magnet { magnetization: _ } => 0b00010001,
            ModelType::Charged { charge: _ } => 0b00001001,
            ModelType::ChargedECR { charge: _ } => 0b00000101,
        };

        self.kernel_voxelize_mesh.set_arg("direction", direction).unwrap();
        self.kernel_voxelize_mesh.set_arg("t", self.t+1).unwrap();
        self.kernel_voxelize_mesh.set_arg("flag", flag).unwrap();
        self.kernel_voxelize_mesh.set_arg("p0", &self.p0).unwrap();
        self.kernel_voxelize_mesh.set_arg("p1", &self.p1).unwrap();
        self.kernel_voxelize_mesh.set_arg("p2", &self.p2).unwrap();
        self.kernel_voxelize_mesh.set_arg("bbu", &self.bbu).unwrap();
        if self.cfg.ext_magneto_hydro {
            match ctype {
                ModelType::Magnet { magnetization } => {
                    self.kernel_voxelize_mesh.set_arg("mpc_x", self.cfg.units.magnetization_si_lu(magnetization.0)).unwrap();
                    self.kernel_voxelize_mesh.set_arg("mpc_y", self.cfg.units.magnetization_si_lu(magnetization.1)).unwrap();
                    self.kernel_voxelize_mesh.set_arg("mpc_z", self.cfg.units.magnetization_si_lu(magnetization.2)).unwrap();
                },
                ModelType::Charged { charge } => {
                    self.kernel_voxelize_mesh.set_arg("mpc_x", self.cfg.units.charge_si_lu(charge)).unwrap();
                }
                _ => {}
            }
            self.kernel_voxelize_mesh.set_arg("tmpF", self.b_dyn.as_ref().expect("msg")).unwrap();
        }
        unsafe {
            self.kernel_voxelize_mesh.cmd().global_work_size(a[direction as usize]).enq().unwrap();
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

// Multiply matrix with matrix
impl Mul<F32_3_3> for F32_3_3{
    type Output = F32_3_3;

    fn mul(self, m: F32_3_3) -> Self::Output {
        F32_3_3 {
            xx: self.xx*m.xx+self.xy*m.yx+self.xz*m.zx, xy: self.xx*m.xy+self.xy*m.yy+self.xz*m.zy, xz: self.xx*m.xz+self.xy*m.yz+self.xz*m.zz,
			yx: self.yx*m.xx+self.yy*m.yx+self.yz*m.zx, yy: self.yx*m.xy+self.yy*m.yy+self.yz*m.zy, yz: self.yx*m.xz+self.yy*m.yz+self.yz*m.zz,
			zx: self.zx*m.xx+self.zy*m.yx+self.zz*m.zx, zy: self.zx*m.xy+self.zy*m.yy+self.zz*m.zy, zz: self.zx*m.xz+self.zy*m.yz+self.zz*m.zz
        }
    }
}
