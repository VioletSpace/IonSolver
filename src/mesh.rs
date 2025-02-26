use crate::lbm::Lbm;
use std::ops::{Add, AddAssign, Mul, Sub};

enum ImpType {
    Solid,
    Magnet,
    Charged,
    ChargedECR,
}

#[derive(Debug, Default, Clone, Copy)]
/// Vector type
struct F32_3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Default, Clone, Copy)]
/// Matrix type
struct F32_3_3 {
    xx: f32,
    yx: f32,
    zx: f32,
    xy: f32,
    yy: f32,
    zy: f32,
    xz: f32,
    yz: f32,
    zz: f32,
}

#[allow(unused)]
struct Mesh {
    triangle_number: u32,
    center: F32_3,
    p_min:  F32_3,
    p_max:  F32_3,
    p0: Vec<F32_3>,
    p1: Vec<F32_3>,
    p2: Vec<F32_3>,
}

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

    fn update_bounds(mut self) {
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

    fn scale(mut self, scale: f32) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] = scale*(self.p0[i]-self.center)+self.center;
			self.p1[i] = scale*(self.p1[i]-self.center)+self.center;
			self.p2[i] = scale*(self.p2[i]-self.center)+self.center;
		}
		self.p_min = scale*(self.p_min-self.center)+self.center;
		self.p_max = scale*(self.p_max-self.center)+self.center;
    }

    fn translate(mut self, translation: F32_3) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] += translation;
			self.p1[i] += translation;
			self.p2[i] += translation;
		}
		self.center += translation;
		self.p_min += translation;
		self.p_max += translation;
    }

    fn rotate(mut self, rotation: F32_3_3) {
        for i in 0..self.triangle_number as usize {
			self.p0[i] = rotation*(self.p0[i]-self.center)+self.center;
			self.p1[i] = rotation*(self.p1[i]-self.center)+self.center;
			self.p2[i] = rotation*(self.p2[i]-self.center)+self.center;
		}
		self.update_bounds();
    }

    fn read_stl() -> Mesh {

        Mesh::new(0, F32_3::default())
    }
}

impl Lbm {
    // Import an STL model as a solid
    pub fn import_solid(mut self, path: &str) {

    }
    
    // Import an STL model as a solid permanent magnet
    pub fn import_magnet(mut self) {
        
    }
    
    // Import an STL model as a solid that is charged
    pub fn import_charged() {
        
    }
    
    // Import a model as a solid that periodically swaps charges for ECR
    pub fn import_charged_() {
        
    }

    fn import_model() {

    }
}

impl Add<F32_3> for F32_3{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        F32_3 {x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z}
    }
}

impl Sub<F32_3> for F32_3{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        F32_3 {x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z}
    }
}

impl Mul<f32> for F32_3{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        F32_3 {x: self.x * rhs, y: self.y * rhs, z: self.z * rhs}
    }
}

impl Mul<F32_3> for f32{
    type Output = F32_3;

    fn mul(self, rhs: F32_3) -> Self::Output {
        F32_3 {x: self * rhs.x, y: self * rhs.y, z: self * rhs.z}
    }
}

impl AddAssign<F32_3> for F32_3{

    fn add_assign(&mut self, rhs: F32_3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Mul<F32_3> for F32_3_3{
    type Output = F32_3;

    fn mul(self, rhs: F32_3) -> Self::Output {
        F32_3 {x: self * rhs.x, y: self * rhs.y, z: self * rhs.z}
    }
}