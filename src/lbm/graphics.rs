//! # graphics
//! 
//! Provides graphics functionality for IonSolver.
//! 
//! This module contains functions to configure the included graphics engine and to render simulations.

use std::{f32::consts::PI, fs::File, io::BufWriter, sync::mpsc::Sender, thread};
use log::info;
use ocl::{Buffer, Kernel, Program, Queue};
use ocl_macros::*;
use crate::{lbm::*, SimState};


/// The vector field used in visualizations like field and streamline
#[allow(unused)]
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum VecVisMode {
    U,
    EStat,
    BStat,
    EDyn,
    BDyn,
}

impl std::fmt::Display for VecVisMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VecVisMode::U => write!(f, "Velocity"),
            VecVisMode::EStat => write!(f, "Static E field"),
            VecVisMode::BStat => write!(f, "Static B field"),
            VecVisMode::EDyn => write!(f, "Dynamic E field"),
            VecVisMode::BDyn => write!(f, "Dynamic B field"),
        }
    }
}

/// The vector field used in visualizations like field and streamline
#[allow(unused)]
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum SliceMode {
    Off,
    X,
    Y,
    Z,
    XZ,
    XYZ,
    YZ,
    XY,
}

impl std::fmt::Display for SliceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceMode::Off => write!(f, "Off"),
            SliceMode::X => write!(f, "X"),
            SliceMode::Y => write!(f, "Y"),
            SliceMode::Z => write!(f, "Z"),
            SliceMode::XZ => write!(f, "X + Z"),
            SliceMode::XYZ => write!(f, "X + Y + Z"),
            SliceMode::YZ => write!(f, "Y + Z"),
            SliceMode::XY => write!(f, "X + Y"),
        }
    }
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum FieldVisMode {
    Vector,
    Scalar,
}

impl std::fmt::Display for FieldVisMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldVisMode::Scalar => write!(f, "Scalar"),
            FieldVisMode::Vector => write!(f, "Vector")
        }
    }
}

/// Keyframe struct. Holds render interval keyframe
#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Keyframe {
    pub time: u64,
    pub repeat: bool,
    pub cam_zoom: f32,
    pub cam_rot_x: f32,
    pub cam_rot_y: f32,
    pub streamline_mode: bool, // Active graphics modes for keyframe
    pub field_mode: bool,
    pub q_mode: bool,
    pub q_field_mode: bool,
    pub flags_mode: bool,
    pub flags_surface_mode: bool,
}

/// Bundles arguments for graphics initialization
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphicsConfig {
    /// Is the graphics engine active (default: false)
    pub graphics_active: bool,
    /// Background color (default: 0x000000 (black))
    pub background_color: u32,
    /// Camera width (default: 1920)
    pub camera_width: u32,
    /// Camera height (default: 1080)
    pub camera_height: u32,
    /// Velocity visualization maximum (default: 0.25)
    pub u_max: f32,
    /// Vorticity visualization minimum (default: 0.0001)
    pub q_min: f32,
    /// Force visualization maximum (default: 0.002)
    pub f_max: f32,
    /// Density
    pub rho_delta: f32,
    /// Draw streamlines every (x) cells (default: 4)
    pub streamline_every: u32,
    /// Length of streamlines (default: 128)
    pub stream_line_lenght: u32,
    /// The vector field used in visualizations like field and streamline (default: U)
    pub vec_vis_mode: VecVisMode,
    pub slice_mode: SliceMode,
    pub slice_x: u32,
    pub slice_y: u32,
    pub slice_z: u32,
    pub max_slice_x: u32,
    pub max_slice_y: u32,
    pub max_slice_z: u32,
    pub field_vis: FieldVisMode,

    /// Visualize a vector field as streamlines
    pub streamline_mode: bool, // Active graphics modes
    /// Visualize a vector field as individual vector lines
    pub field_mode: bool,
    /// Visualize a vector or density field as a slice
    pub field_slice_mode: bool,
    /// Visualize vorticity
    pub q_mode: bool,
    pub q_field_mode: bool,
    /// Visualize simulation flags as wireframe
    pub flags_mode: bool,
    /// Visualize simulation flags with marching cubes surface reconstruction
    pub flags_surface_mode: bool,
    /// Visualize coordinate system axes
    pub axes_mode: bool,
    /// Visualize where the ECR condition is met
    pub ecrc_mode: bool,

    // Configure rendering keyframed intervals with camera positions
    /// Activate rendering intervals:
    pub render_intervals: bool,
    pub keyframes: Vec<Keyframe>,
}

impl GraphicsConfig {
    pub fn new() -> GraphicsConfig {
        GraphicsConfig {
            graphics_active: true,
            background_color: 0x000000,
            camera_width: 1920,
            camera_height: 1080,
            u_max: 0.25,
            q_min: 0.0001,
            f_max: 0.002,
            rho_delta: 0.5,
            streamline_every: 4,
            stream_line_lenght: 128,
            vec_vis_mode: VecVisMode::U,
            slice_mode: SliceMode::Off,
            slice_x: 0,
            slice_y: 0,
            slice_z: 0,
            max_slice_x: 0,
            max_slice_y: 0,
            max_slice_z: 0,
            field_vis: FieldVisMode::Vector,

            streamline_mode: false,
            field_mode: false,
            field_slice_mode: false,
            q_mode: false,
            q_field_mode: false,
            flags_mode: false,
            flags_surface_mode: false,
            axes_mode: false,
            ecrc_mode: false,
            render_intervals: false,
            keyframes: vec![],
        }
    }
}

/// LbmDomain Graphics struct used to render itself to a color and z buffer.
///
/// Each LbmDomain renders its own frame. Different domain frames are stitched back together in the Lbm draw_frame function.
pub struct Graphics {
    kernel_clear: Kernel,
    pub bitmap: Buffer<i32>,
    pub zbuffer: Buffer<i32>,
    pub camera_params: Buffer<f32>,

    kernel_graphics_axes: Kernel,
    kernel_graphics_field: Kernel,
    kernel_graphics_field_slice: Kernel,
    kernel_graphics_flags: Kernel,
    kernel_graphics_flags_mc: Kernel,
    kernel_graphics_q: Kernel,
    kernel_graphics_q_field: Kernel,
    kernel_graphics_streamline: Kernel,
    kernel_graphics_ecrc_met: Option<Kernel>,
}

impl Graphics {
    #[rustfmt::skip]
    pub fn new(
        lbm_config: &mut LbmConfig,
        program: &Program,
        queue: &Queue,
        flags: &Buffer<u8>,
        u: &Buffer<f32>,
        rho: &Buffer<f32>,
        b: &Option<Buffer<f32>>,
        n_d: (u32, u32, u32),
    ) -> Graphics {
        let width =  lbm_config.graphics_config.camera_width;
        let height = lbm_config.graphics_config.camera_height;
        lbm_config.graphics_config.max_slice_x = lbm_config.n_x - 1;
        lbm_config.graphics_config.max_slice_y = lbm_config.n_y - 1;
        lbm_config.graphics_config.max_slice_z = lbm_config.n_z - 1;
        let n = n_d.0 as u64 * n_d.1 as u64 * n_d.2 as u64;

        info!("Allocating graphics buffers...");
        let bitmap =        buffer!(queue, [width, height], 0i32);
        let zbuffer =       buffer!(queue, [width, height], 0i32);
        let camera_params = buffer!(queue, 15, 0.0f32);
        bwrite!(camera_params, new_camera_params());

        let sln = match lbm_config.velocity_set {
            VelocitySet::D2Q9 => (lbm_config.n_x / lbm_config.graphics_config.streamline_every) as u64 * (lbm_config.n_y / lbm_config.graphics_config.streamline_every) as u64,
            _ => (lbm_config.n_x / lbm_config.graphics_config.streamline_every) as u64 * (lbm_config.n_y / lbm_config.graphics_config.streamline_every) as u64 * (lbm_config.n_z / lbm_config.graphics_config.streamline_every) as u64,
        };

        info!("Initializing graphics kernels...");
        // Clear kernel
        let kernel_clear = kernel!(program, queue, "graphics_clear", bitmap.len(), ("bitmap", &bitmap), ("zbuffer", &zbuffer));

        // Graphics/Visualization kernels:
        let kernel_graphics_axes       = kernel!(program, queue, "graphics_axes",           1,                                           ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer));
        let kernel_graphics_field      = kernel!(program, queue, "graphics_field",        [n], ("flags", flags), ("u", u),               ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer),                    ("slice_mode", 0), ("slice_x", 0), ("slice_y", 0), ("slice_z", 0));
        let kernel_graphics_field_slice= kernel!(program, queue, "graphics_field_slice",  [1], ("flags", flags), ("u", u), ("rho", rho), ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer), ("field_mode", 0), ("slice_mode", 0), ("slice_x", 0), ("slice_y", 0), ("slice_z", 0));
        let kernel_graphics_flags      = kernel!(program, queue, "graphics_flags",        [n], ("flags", flags),                         ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer));
        let kernel_graphics_flags_mc   = kernel!(program, queue, "graphics_flags_mc",     [n], ("flags", flags),                         ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer));
        let kernel_graphics_q          = kernel!(program, queue, "graphics_q",            [n], ("flags", flags), ("u", u),               ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer));
        let kernel_graphics_q_field    = kernel!(program, queue, "graphics_q_field",      [n], ("flags", flags), ("u", u),               ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer));
        let kernel_graphics_streamline = kernel!(program, queue, "graphics_streamline", [sln], ("flags", flags), ("u", u),               ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer),                    ("slice_mode", 0), ("slice_x", 0), ("slice_y", 0), ("slice_z", 0));
        let mut kernel_graphics_ecrc_met = None;

        if lbm_config.ext_magneto_hydro {
            kernel_graphics_ecrc_met = Some(kernel!(program, queue, "graphics_ecrc_met", [n], ("flags", flags), ("B", b.as_ref().expect("B")), ("ecrf", lbm_config.ecr_freq), ("camera_params", &camera_params), ("bitmap", &bitmap), ("zbuffer", &zbuffer)))
        }

        Graphics {
            kernel_clear,
            bitmap,
            zbuffer,
            camera_params,

            kernel_graphics_axes,
            kernel_graphics_field,
            kernel_graphics_field_slice,
            kernel_graphics_flags,
            kernel_graphics_flags_mc,
            kernel_graphics_q,
            kernel_graphics_q_field,
            kernel_graphics_streamline,
            kernel_graphics_ecrc_met,
        }
    }
}

// draw_frame function for Lbm
impl super::Lbm {
    #[allow(unused_variables)]
    pub fn draw_frame(&self, save: bool, name: String, sim_tx: Sender<SimState>, i: &u32) {
        let width = self.config.graphics_config.camera_width;
        let height = self.config.graphics_config.camera_height;
        let domain_numbers = self.get_d_n();
        let mut bitmap: Vec<i32> = vec![0; (width * height) as usize]; // Base bitmap
        let mut zbuffer: Vec<i32> = vec![0; (width * height) as usize];
        let mut bitmaps: Vec<Vec<i32>> =
            vec![vec![0; (width * height) as usize]; domain_numbers - 1]; // Holds later domain bitmaps
        let mut zbuffers: Vec<Vec<i32>> =
            vec![vec![0; (width * height) as usize]; domain_numbers - 1];
        for d in 0..domain_numbers {
            self.domains[d].enqueue_draw_frame(self.config.graphics_config.clone());
        }
        for d in 0..domain_numbers {
            self.domains[d].queue.finish().unwrap();

            if d == 0 {
                bread!(self.domains[d].graphics.as_ref().expect("graphics").bitmap, bitmap);
                bread!(self.domains[d].graphics.as_ref().expect("graphics").zbuffer, zbuffer);
            } else {
                bread!(self.domains[d].graphics.as_ref().expect("graphics").bitmap, bitmaps[d - 1]);
                bread!(self.domains[d].graphics.as_ref().expect("graphics").zbuffer, zbuffers[d - 1]);
            }
        }
        self.finish_queues();

        let i = *i;
        thread::spawn(move || {
            // Generating images needs own thread for performance reasons
            for d in 0..domain_numbers - 1 {
                let bitmap_d = &bitmaps[d];
                let zbuffer_d = &zbuffers[d];
                for i in 0..(width * height) as usize {
                    let zdi = zbuffer_d[i];
                    if zdi > zbuffer[i] {
                        bitmap[i] = bitmap_d[i];
                        zbuffer[i] = zbuffer_d[i];
                    }
                }
            }

            let mut save_buffer: Vec<u8> = Vec::with_capacity(bitmap.len());

            #[cfg(feature = "gui")]
            let mut pixels: Vec<egui::Color32> = Vec::with_capacity(bitmap.len());

            for pixel in &bitmap {
                let color = pixel & 0xFFFFFF;

                #[cfg(feature = "gui")]
                pixels.push(egui::Color32::from_rgb(
                    ((color >> 16) & 0xFF) as u8,
                    ((color >> 8) & 0xFF) as u8,
                    (color & 0xFF) as u8,
                ));
                if save {
                    // only update save buffer if required
                    save_buffer.push(((color >> 16) & 0xFF) as u8);
                    save_buffer.push(((color >> 8) & 0xFF) as u8);
                    save_buffer.push((color & 0xFF) as u8);
                }
            }

            #[cfg(feature = "gui")]
            let color_image = egui::ColorImage {
                size: [width as usize, height as usize],
                pixels,
            };

            #[cfg(feature = "gui")] {
                _ = sim_tx.send(SimState {
                    step: i,
                    img: Some(color_image),
                    graphics_cfg: None,
                }); // This may fail if the simulation is terminated, but a frame is still being generated. Error can be ignored.
            }
            

            if save {
                thread::spawn(move || {
                    //Saving needs own thread for performance reasons
                    let file = File::create(format!(r"out/{}_{}.png", name, i)).unwrap();
                    let ref mut w = BufWriter::new(file);
                    let mut encoder = png::Encoder::new(w, width, height);
                    encoder.set_color(png::ColorType::Rgb);
                    let mut writer = encoder.write_header().unwrap();
                    writer.write_image_data(&save_buffer).unwrap();
                });
            }
        });
    }

    // Render keyframes
    pub fn render_keyframes(&self, sim_tx: Sender<SimState>, i: &u32) {
        if self.config.graphics_config.render_intervals {
            for (c, frame) in self.config.graphics_config.keyframes.iter().enumerate() {
                if (frame.time % (i+1) as u64 == 0 && frame.repeat) || (frame.time == *i as u64 && !frame.repeat) {
                    let mut params = graphics::camera_params_rot(
                        frame.cam_rot_x * (PI / 180.0),
                        frame.cam_rot_y * (PI / 180.0),
                    );
                    params[0] = frame.cam_zoom;
                    for d in &self.domains {
                        bwrite!(d.graphics.as_ref().expect("graphics").camera_params, params);
                    }
                    self.draw_frame(true,  format!("frame_{}", c), sim_tx.clone(), i)
                }
            }
        }
    }
}

// enqueue_draw_frame function for LbmDomain
#[rustfmt::skip]
impl domain::LbmDomain {
    pub fn enqueue_draw_frame(&self, cfg: GraphicsConfig) {
        let graphics = self
            .graphics
            .as_ref()
            .expect("Graphics used but not initialized");
        // Kernel enqueueing is unsafe
        unsafe {
            graphics.kernel_clear.enq().unwrap();
            if cfg.axes_mode {
                graphics.kernel_graphics_axes.enq().unwrap();
            }
            if cfg.streamline_mode {
                // Streamlines can show velocity, dynamic and static EStat and BStat field
                graphics.kernel_graphics_streamline.set_arg("u", match cfg.vec_vis_mode {
                    VecVisMode::U => &self.u,
                    VecVisMode::EStat => self.e_stat.as_ref().expect("E_stat buffer used but not initialized"),
                    VecVisMode::BStat => self.b_stat.as_ref().expect("B_stat buffer used but not initialized"),
                    VecVisMode::EDyn => self.e_dyn.as_ref().expect("E_dyn buffer used but not initialized"),
                    VecVisMode::BDyn => self.b_dyn.as_ref().expect("B_dyn buffer used but not initialized"),
                }).unwrap();
                graphics.kernel_graphics_streamline.set_arg("slice_mode", cfg.slice_mode as u32).unwrap();
                graphics.kernel_graphics_streamline.set_arg("slice_x", cfg.slice_x).unwrap();
                graphics.kernel_graphics_streamline.set_arg("slice_y", cfg.slice_y).unwrap();
                graphics.kernel_graphics_streamline.set_arg("slice_z", cfg.slice_z).unwrap();
                graphics.kernel_graphics_streamline.enq().unwrap();
            }
            if cfg.field_mode {
                graphics.kernel_graphics_field.set_arg("u", match cfg.vec_vis_mode {
                    VecVisMode::U => &self.u,
                    VecVisMode::EStat => self.e_stat.as_ref().expect("E_stat buffer used but not initialized"),
                    VecVisMode::BStat => self.b_stat.as_ref().expect("B_stat buffer used but not initialized"),
                    VecVisMode::EDyn => self.e_dyn.as_ref().expect("E_dyn buffer used but not initialized"),
                    VecVisMode::BDyn => self.b_dyn.as_ref().expect("B_dyn buffer used but not initialized"),
                }).unwrap();
                graphics.kernel_graphics_field.set_arg("slice_mode", cfg.slice_mode as u32).unwrap();
                graphics.kernel_graphics_field.set_arg("slice_x", cfg.slice_x).unwrap();
                graphics.kernel_graphics_field.set_arg("slice_y", cfg.slice_y).unwrap();
                graphics.kernel_graphics_field.set_arg("slice_z", cfg.slice_z).unwrap();
                graphics.kernel_graphics_field.enq().unwrap();
            }
            if cfg.field_slice_mode {
                graphics.kernel_graphics_field_slice.set_arg("u", match cfg.vec_vis_mode {
                    VecVisMode::U => &self.u,
                    VecVisMode::EStat => self.e_stat.as_ref().expect("E_stat buffer used but not initialized"),
                    VecVisMode::BStat => self.b_stat.as_ref().expect("B_stat buffer used but not initialized"),
                    VecVisMode::EDyn => self.e_dyn.as_ref().expect("E_dyn buffer used but not initialized"),
                    VecVisMode::BDyn => self.b_dyn.as_ref().expect("B_dyn buffer used but not initialized"),
                }).unwrap();
                graphics.kernel_graphics_field_slice.set_arg("field_mode", cfg.field_vis as u32).unwrap();
                graphics.kernel_graphics_field_slice.set_arg("slice_mode", cfg.slice_mode as u32).unwrap();
                graphics.kernel_graphics_field_slice.set_arg("slice_x", cfg.slice_x).unwrap();
                graphics.kernel_graphics_field_slice.set_arg("slice_y", cfg.slice_y).unwrap();
                graphics.kernel_graphics_field_slice.set_arg("slice_z", cfg.slice_z).unwrap();
                match cfg.slice_mode {
                    SliceMode::Off => {},
                    SliceMode::X | SliceMode::Y | SliceMode::Z => {
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(cfg.slice_mode as u32 - 1)).enq().unwrap();
                    },
                    SliceMode::XZ => {
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 0u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(0)).enq().unwrap();
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 2u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(2)).enq().unwrap();
                    },
                    SliceMode::XYZ => {
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 0u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(0)).enq().unwrap();
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 1u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(1)).enq().unwrap();
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 2u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(2)).enq().unwrap();
                    },
                    SliceMode::YZ => {
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 1u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(1)).enq().unwrap();
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 2u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(2)).enq().unwrap();
                    },
                    SliceMode::XY => {
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 0u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(0)).enq().unwrap();
                        graphics.kernel_graphics_field_slice.set_arg("slice_mode", 1u32+1).unwrap();
                        graphics.kernel_graphics_field_slice.cmd().global_work_size(self.get_area(1)).enq().unwrap();
                    },
                }
                
            }
            if cfg.q_mode {
                graphics.kernel_graphics_q.enq().unwrap();
            }
            if cfg.q_field_mode {
                graphics.kernel_graphics_q_field.enq().unwrap();
            }
            if cfg.flags_mode {
                graphics.kernel_graphics_flags.enq().unwrap();
            }
            if cfg.flags_surface_mode {
                graphics.kernel_graphics_flags_mc.enq().unwrap();
            }
            if cfg.ecrc_mode {
                if self.cfg.ext_magneto_hydro {
                    graphics.kernel_graphics_ecrc_met.as_ref().expect("kernel").enq().unwrap();
                }
            }
        }
    }
}

#[rustfmt::skip]
/// Returns a string of OpenCL C `#define`s from the provided arguments that are appended to the base OpenCl code at runtime.
pub fn get_graphics_defines(graphics_config: &GraphicsConfig) -> String {
    "\n	#define GRAPHICS".to_owned()
    +"\n	#define DEF_BACKGROUND_COLOR "  + &graphics_config.background_color.to_string()
    +"\n	#define DEF_SCREEN_WIDTH "      + &graphics_config.camera_width.to_string()+"u"
    +"\n	#define DEF_SCREEN_HEIGHT "     + &graphics_config.camera_height.to_string()+"u"
    +"\n	#define DEF_SCALE_U "           + &format!("{:?}f", 1.0f32 / (0.57735027f32 * graphics_config.u_max))
    +"\n	#define DEF_SCALE_RHO "         + &format!("{:?}f", 0.5f32 / graphics_config.rho_delta)
    +"\n	#define DEF_SCALE_Q_MIN "       + &graphics_config.q_min.to_string()+"f"
    +"\n	#define DEF_SCALE_F "           + &(1.0f32 / graphics_config.f_max).to_string()+"f"
    +"\n	#define DEF_STREAMLINE_SPARSE " + &graphics_config.streamline_every.to_string()+"u"
    +"\n	#define DEF_STREAMLINE_LENGTH " + &graphics_config.stream_line_lenght.to_string()+"u"
    +"\n	#define COLOR_S (127<<16|127<<8|127)"
    +"\n	#define COLOR_E (  0<<16|255<<8|  0)"
    +"\n	#define COLOR_M (255<<16|  0<<8|255)"
    +"\n	#define COLOR_T (255<<16|  0<<8|  0)"
    +"\n	#define COLOR_F (  0<<16|  0<<8|255)"
    +"\n	#define COLOR_I (  0<<16|255<<8|255)"
    +"\n	#define COLOR_0 (127<<16|127<<8|127)"
    +"\n	#define COLOR_X (255<<16|127<<8|  0)"
    +"\n	#define COLOR_Y (255<<16|255<<8|  0)"
    +"\n	#define COLOR_P (255<<16|255<<8|191)"
}

pub fn new_camera_params() -> Vec<f32> {
    let mut params: Vec<f32> = vec![0.0; 15];
    //Defaults from FluidX3D: graphics.hpp:20
    let eye_dist: u32 = 0;

    let rx = 0.5 * PI;
    let sinrx = rx.sin();
    let cosrx = rx.cos();
    let ry = PI;
    let sinry = ry.sin();
    let cosry = ry.cos();

    params[0] = 3.0; //zoom
    params[1] = 512.0; //distance from rotation center
                       //2-4 is pos x y z
                       //5-13 is a rotation matrix
    params[5] = cosrx; //Rxx
    params[6] = sinrx; //Rxy
    params[7] = 0.0; //Rxz
    params[8] = sinrx * sinry; //Ryx
    params[9] = -cosrx * sinry; //Ryy
    params[10] = cosry; //Ryz
    params[11] = -sinrx * cosry; //Rzx
    params[12] = cosrx * cosry; //Rzy
    params[13] = sinry; //Rzz
    params[14] = ((false as u32) << 31 | (false as u32) << 30 | (eye_dist & 0xFFFF)) as f32;
    params
}

pub fn camera_params_rot(rx: f32, ry: f32) -> Vec<f32> {
    let mut params: Vec<f32> = vec![0.0; 15];
    //Defaults from FluidX3D: graphics.hpp:20
    let eye_dist: u32 = 0;

    //let rx = 0.5 * PI;
    let sinrx = rx.sin();
    let cosrx = rx.cos();
    //let ry = PI;
    let sinry = ry.sin();
    let cosry = ry.cos();

    params[0] = 3.0; //zoom
    params[1] = 512.0; //distance from rotation center
                       //2-4 is pos x y z
                       //5-13 is a rotation matrix
    params[5] = cosrx; //Rxx
    params[6] = sinrx; //Rxy
    params[7] = 0.0; //Rxz
    params[8] = sinrx * sinry; //Ryx
    params[9] = -cosrx * sinry; //Ryy
    params[10] = cosry; //Ryz
    params[11] = -sinrx * cosry; //Rzx
    params[12] = cosrx * cosry; //Rzy
    params[13] = sinry; //Rzz
    params[14] = ((false as u32) << 31 | (false as u32) << 30 | (eye_dist & 0xFFFF)) as f32;
    params
}
