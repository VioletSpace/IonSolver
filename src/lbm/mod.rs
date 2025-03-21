//! # lbm
//! 
//! Contains methods for creating and running magnetohydrodynamic LBM simulations
//! 
//! Structured into:
//! - `lbm`: LBM configuration structs, LBM simulation master struct, LbmDomain structs for single devices
//! - `mod graphics`: Graphics functionality for visualization
//! - `mod multi-node`: Extensions for simulation execution on multiple compute nodes (optional)
//! - `mod precompute`: Precomputation of electric and magnetic fields for simulation start up
//! - `mod units`: Units struct for unit conversion between simulation and SI units
//! 
//! This is the core funtionality needed for IonSolver.

use std::cmp;

mod domain;
pub mod graphics;
#[cfg(feature = "multi-node")]
pub mod multi_node;
mod units;
mod types;

use crate::opencl;
pub use {domain::LbmDomain, graphics::GraphicsConfig};
use log::{info, warn};
pub use types::*;
pub use units::Units;
use crate::mesh::Mesh;

// Helper Functions
/// Get `x, y, z` coordinates from 1D index `n` and side lengths `n_x` and `n_y`.
fn get_coordinates_sl(n: u64, n_x: u32, n_y: u32) -> (u32, u32, u32) {
    let t: u64 = n % (n_x as u64 * n_y as u64);
    //x, y, z
    (
        (t % n_x as u64) as u32,
        (t / n_x as u64) as u32,
        (n / (n_x as u64 * n_y as u64)) as u32,
    )
}

/// Struct used to bundle arguments for LBM simulation setup.
/// 
/// Use this struct to configure a simulation and then instantiate the simulation with `Lbm::new(LbmConfig)`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LbmConfig {
    // Holds init information about the simulation
    /// Velocity discretization mode
    pub velocity_set: VelocitySet,
    /// Simulation relaxation time type
    pub relaxation_time: RelaxationTime,
    /// Simulation float type for memory compression of simulation data (DDFs)
    pub float_type: FloatType,
    /// Struct for unit conversion
    pub units: Units,
    /// Size of simulation on each axis
    pub n_x: u32,
    pub n_y: u32,
    pub n_z: u32,

    /// Number of domains on each axis
    pub d_x: u32,
    pub d_y: u32,
    pub d_z: u32,

    /// Kinematic viscosity
    pub nu: f32,
    /// Force on each axis. Needs ext_volume_force to work
    pub f_x: f32,
    pub f_y: f32,
    pub f_z: f32,

    //Extensions
    /// Enable equilibrium boudaries extension (for inlets/outlets)
    pub ext_equilibrium_boudaries: bool,
    /// Enable volume force extension (for handling of complex forces)
    pub ext_volume_force: bool,
    /// Enable force field extension (i. E. for gravity). Needs ext_volume_force to work
    pub ext_force_field: bool,
    /// Enable magnetohydrodynamics extension. Needs ext_volume_force to work 
    pub ext_magneto_hydro: bool,
    /// Enable the subgrid ECR extension to model small-scale electron cyclotron resonance. Needs ext_magneto_hydro to work 
    pub ext_subgrid_ecr: bool,

    /// LOD option for dynamic fields.
    /// 
    /// Dynamic field quality improves with higher values: 1 is very coarse, 4 is higher quality (Performance does not peek at lowest values).
    /// Set this to 0 to disable LODs (VERY SLOW). 
    pub mhd_lod_depth: u8,

    /// ECR frequency
    pub ecr_freq: f32,
    /// ECR field strenght
    pub ecr_field_strength: f32,

    /// Configuration struct for the built-in graphics engine
    pub graphics_config: GraphicsConfig,

    /// Run the simulation for x steps
    pub run_steps: u64,
}

impl LbmConfig {
    /// Returns `LbmConfig` with default values
    pub fn new() -> LbmConfig {
        LbmConfig {
            velocity_set: VelocitySet::D2Q9,
            relaxation_time: RelaxationTime::Srt,
            float_type: FloatType::FP16S,
            units: units::Units::new(),
            n_x: 1,
            n_y: 1,
            n_z: 1,
            d_x: 1,
            d_y: 1,
            d_z: 1,
            nu: 1.0f32 / 6.0f32,
            f_x: 0.0f32,
            f_y: 0.0f32,
            f_z: 0.0f32,

            ext_equilibrium_boudaries: false,
            ext_volume_force: false,
            ext_force_field: false,
            ext_magneto_hydro: false,
            ext_subgrid_ecr: false,

            mhd_lod_depth: 4, // Dynamic field LODs
            ecr_freq: 0.0f32,
            ecr_field_strength: 0.0f32,

            graphics_config: graphics::GraphicsConfig::new(),
            run_steps: 0,
        }
    }
}

/// To start a LBM simulation, initialise an Lbm struct:
/// ```
/// Lbm::new(lbm_config: LbmConfig)
/// ```
/// The `new()` function takes in another struct, the [`LbmConfig`], which contains all necessary arguments.
/// [`LbmConfig`] needs to be configured beforehand.
///
/// The Lbm struct contains one or multiple [`LbmDomain`] structs.
/// The Simulation is actually run on these [`LbmDomain`] structs, each Domain corresponding to an OpenCL device,
/// enabling multi-device parallelization.
/// [`LbmDomain`] initialisation is handled automatically in `Lbm::new()` using `LbmDomain::new()`
///
/// [`LbmConfig`]: crate::lbm::LbmConfig
/// [`LbmDomain`]: crate::lbm::LbmDomain
pub struct Lbm {
    /// Vector of [`LbmDomain`]s that are part of this simulation.
    pub domains: Vec<LbmDomain>,
    /// A copy of the [`LbmConfig`] used to initialize the simulation.
    pub config: LbmConfig,
    /// Imported meshes
    pub meshes: Vec<Mesh>,
    initialized: bool,
}

impl Lbm {
    // TODO: add read for charges and magnets
    /// Returns new `Lbm` struct from pre-configured `LbmConfig` struct. `LbmDomain` setup is handled automatically.
    /// Configures Domains
    pub fn new(mut lbm_config: LbmConfig) -> Lbm {
        let n_d_x: u32 = (lbm_config.n_x / lbm_config.d_x) * lbm_config.d_x;
        let n_d_y: u32 = (lbm_config.n_y / lbm_config.d_y) * lbm_config.d_y;
        let n_d_z: u32 = (lbm_config.n_z / lbm_config.d_z) * lbm_config.d_z;
        if n_d_x != lbm_config.n_x || n_d_y != lbm_config.n_y || n_d_z != lbm_config.n_z {
            warn!(
                "Resolution {}, {}, {} not divisible by Domains: Overiding resolution.",
                lbm_config.n_x, lbm_config.n_y, lbm_config.n_z
            )
        }

        lbm_config.n_x = n_d_x;
        lbm_config.n_y = n_d_y;
        lbm_config.n_z = n_d_z;

        let domain_numbers: u32 = lbm_config.d_x * lbm_config.d_y * lbm_config.d_z;

        let device_infos = opencl::device_selection(domain_numbers);
        //TODO: sanity check

        let mut lbm_domains: Vec<LbmDomain> = Vec::new();
        for d in 0..domain_numbers {
            info!("Initializing domain {}/{}", d + 1, domain_numbers);
            let x = (d % (lbm_config.d_x * lbm_config.d_y)) % lbm_config.d_x;
            let y = (d % (lbm_config.d_x * lbm_config.d_y)) / lbm_config.d_x;
            let z = d / (lbm_config.d_x * lbm_config.d_y);
            info!("Using \"{}\" for domain {}", device_infos[d as usize].name().unwrap(), d + 1);
            lbm_domains.push(LbmDomain::new(
                &mut lbm_config,
                device_infos[d as usize],
                x,
                y,
                z,
                d,
            ))
        }
        info!("All domains initialized.\n");

        Lbm {
            domains: lbm_domains,
            config: lbm_config,
            meshes: vec![],
            initialized: false,
        }
    }

    /// Readies the LBM Simulation to be run.
    /// Executes `kernel_initialize` Kernels for every `LbmDomain` and fills domain transfer buffers.
    pub fn initialize(&mut self) {
        // the communicate calls at initialization need an odd time step
        self.increment_timestep(1);
        self.communicate_rho_u_flags();
        self.kernel_initialize();
        self.communicate_rho_u_flags();
        self.communicate_fi();
        if self.config.ext_magneto_hydro {
            self.communicate_fqi();
            self.communicate_ei();
            self.communicate_qu_lods();
            self.update_e_b_dynamic();
        }

        self.finish_queues();
        self.reset_timestep();
        self.initialized = true;
    }

    /// Runs Simulations for `steps`
    #[allow(unused)]
    pub fn run(&mut self, steps: u64) {
        //Initialize, then run simulation for steps
        if !self.initialized {
            //Run initialization Kernel
            self.initialize();
        }
        for i in 0..steps {
            //println!("Step {}", i);
            self.do_time_step();
        }
    }

    /// Executes one LBM time step.
    /// Executes `kernel_stream_collide` Kernels for every `LbmDomain` and updates domain transfer buffers.
    /// Updates the dynamic E and B fields.
    pub fn do_time_step(&mut self) {
        if self.config.ext_magneto_hydro {
            self.clear_qu_lod(); // Ready LOD Buffer
        }
        // call kernel stream_collide to perform one LBM time step
        self.stream_collide();
        if self.config.graphics_config.graphics_active {
            self.communicate_rho_u_flags();
        }
        self.communicate_fi();
        if self.config.ext_magneto_hydro {
            if self.domains.len() > 1 { self.build_lods_part_2() };
            self.communicate_fqi();
            self.communicate_ei();
            self.communicate_qu_lods();
            self.update_e_b_dynamic();
        }
        if self.get_d_n() == 1 || self.config.ext_magneto_hydro {
            // Additional synchronization only needed in single-GPU or after E&B update
            self.finish_queues();
        }
        self.increment_timestep(1);
    }

    /// Blocks execution until all `LbmDomain` OpenCL queues have finished.
    pub fn finish_queues(&self) {
        for d in 0..self.domains.len() {
            self.domains[d].queue.finish().unwrap();
        }
    }

    /// Computes the static magnetic field from imported magnet type meshes. This is only available with the
    /// magneto_hydro extension and needs to be called before the simulation initialization.
    #[allow(unused)]
    pub fn precompute_B(&mut self) {
        if self.config.ext_magneto_hydro {
            info!("Calculating mesh magnetic field. (This may take a while)");
            let now = std::time::Instant::now();
            for d in &mut self.domains {
                d.enqueue_precompute_b();
            }
            self.finish_queues();
            info!("Calculated mesh magnetic field in {} s", now.elapsed().as_secs());
        } else {
            warn!("Electromagnetic field computation is only available wit the magneto_hydro extension.")
        }
        
    }

    /// Computes the static electric field from imported charged type meshes. This is only available with the
    /// magneto_hydro extension and needs to be called before the simulation initialization.
    #[allow(unused)]
    pub fn precompute_E(&mut self) {
        if self.config.ext_magneto_hydro {
            info!("Calculating mesh electric field. (This may take a while)");
            let now = std::time::Instant::now();
            for d in &mut self.domains {
                d.enqueue_precompute_e();
            }
            self.finish_queues();
            info!("Calculated mesh electric field in {} s", now.elapsed().as_secs());
        } else {
            warn!("Electromagnetic field computation is only available wit the magneto_hydro extension.")
        }
    }

    /// Executes initialize kernel
    fn kernel_initialize(&self) {
        for d in 0..self.get_d_n() {
            self.domains[d].enqueue_initialize().unwrap();
        }
    }

    /// Execute stream_collide kernel on all domains
    fn stream_collide(&self) {
        for d in 0..self.get_d_n() {
            self.domains[d].enqueue_stream_collide().unwrap();
        }
    }

    /// Execute E and B dynamic update kernel on all domains
    fn update_e_b_dynamic(&self) {
        for d in 0..self.get_d_n() {
            self.domains[d].enqueue_update_e_b_dyn().unwrap();
        }
    }

    /// Build lower detail LODs needed for other domains from highest-detail LODs generated in stream_collide
    fn build_lods_part_2(&self) {
        for d in 0..self.get_d_n() {
            self.domains[d].enqueue_lod_part_2_gather().unwrap();
        }
    }

    /// Reset charge and velocity LOD
    fn clear_qu_lod(&self) {
        for d in 0..self.get_d_n() {
            self.domains[d].enqueue_clear_qu_lod().unwrap();
        }
    }

    // Domain communication
    /// Communicate a field across domain barriers
    #[rustfmt::skip]
    fn communicate_field(&mut self, field: TransferField, bytes_per_cell: usize) {
        let d_x = self.config.d_x as usize;
        let d_y = self.config.d_y as usize;
        let d_z = self.config.d_z as usize;
        let d_n = self.get_d_n();

        if d_x > 1 { // Communicate x-axis
            for d in 0..d_n {self.domains[d].enqueue_transfer_extract_field(field, 0, bytes_per_cell).unwrap();} // Extract into transfer buffers
            self.finish_queues(); // Synchronize domains
            for d in 0..d_n { // Swap transfer buffers at domain boundaries
                let (x, y, z) = ((d % (d_x * d_y)) % d_x, (d % (d_x * d_y)) / d_x, d / (d_x * d_y)); // Domain x, y and z coord
                let dxp = ((x + 1) % d_x) + (y + z * d_y) * d_x; // domain index of domain at x+1
                unsafe {std::ptr::swap(&mut self.domains[d].transfer_p_host as *mut _, &mut self.domains[dxp].transfer_m_host as *mut _);} // Swap transfer buffers without copying them
            }
            for d in 0..d_n {self.domains[d].enqueue_transfer_insert_field(field, 0, bytes_per_cell).unwrap();} // Insert from transfer buffers
        }
        if d_y > 1 { // Communicate y-axis
            for d in 0..d_n {self.domains[d].enqueue_transfer_extract_field(field, 1, bytes_per_cell).unwrap();} // Extract into transfer buffers
            self.finish_queues(); // Synchronize domains
            for d in 0..d_n { // Swap transfer buffers at domain boundaries
                let (x, y, z) = ((d % (d_x * d_y)) % d_x, (d % (d_x * d_y)) / d_x, d / (d_x * d_y)); // Domain x, y and z coord
                let dyp = x + (((y + 1) % d_y) + z * d_y) * d_x; // domain index of domain at y+1
                unsafe {std::ptr::swap(&mut self.domains[d].transfer_p_host as *mut _, &mut self.domains[dyp].transfer_m_host as *mut _);} // Swap transfer buffers without copying them
            }
            for d in 0..d_n {self.domains[d].enqueue_transfer_insert_field(field, 1, bytes_per_cell).unwrap();} // Insert from transfer buffers
        }
        if d_z > 1 { // Communicate z-axis
            for d in 0..d_n {self.domains[d].enqueue_transfer_extract_field(field, 2, bytes_per_cell).unwrap();} // Extract into transfer buffers
            self.finish_queues(); // Synchronize domains
            for d in 0..d_n { // Swap transfer buffers at domain boundaries
                let (x, y, z) = ((d % (d_x * d_y)) % d_x, (d % (d_x * d_y)) / d_x, d / (d_x * d_y)); // Domain x, y and z coord
                let dzp = x + (y + ((z + 1) % d_z) * d_y) * d_x; // domain index of domain at z+1
                unsafe {std::ptr::swap(&mut self.domains[d].transfer_p_host as *mut _, &mut self.domains[dzp].transfer_m_host as *mut _);} // Swap transfer buffers without copying them
            }
            for d in 0..d_n {self.domains[d].enqueue_transfer_insert_field(field, 2, bytes_per_cell).unwrap();} // Insert from transfer buffers
        }
    }

    /// Communicate Fi across domain boundaries
    fn communicate_fi(&mut self) {
        let bytes_per_cell =
            self.config.float_type.size_of() * self.config.velocity_set.get_transfers(); // FP type size * transfers.
        self.communicate_field(TransferField::Fi, bytes_per_cell);
    }
    /// Communicate rho, u and flags across domain boundaries (needed for graphics)
    fn communicate_rho_u_flags(&mut self) {
        self.communicate_field(TransferField::RhoUFlags, 17);
    }

    /// Communicate Qi across domain boundaries
    fn communicate_fqi(&mut self) {
        let bytes_per_cell = self.config.float_type.size_of() * 1; // FP type size * transfers. The fixed D3Q7 lattice has 1 transfer
        self.communicate_field(TransferField::Qi, bytes_per_cell);
    }

    /// Communicate Ei across domain boundaries
    fn communicate_ei(&mut self) {
        let bytes_per_cell =
            self.config.float_type.size_of() * self.config.velocity_set.get_transfers(); // FP type size * transfers.
        self.communicate_field(TransferField::Ei, bytes_per_cell);
    }

    /// Communicate charge and velocity LODs across domains
    //#[allow(unused)]
    #[rustfmt::skip]
    fn communicate_qu_lods(&mut self) {
        let d_n = self.get_d_n();
        let dim = self.config.velocity_set.get_set_values().0 as u32;

        fn get_offset(depth: i32, dim: u32) -> usize {
            let mut c = 0;
            for i in 0..=depth {c += ((1<<i) as usize).pow(dim)};
            c
        }

        if d_n > 1 {
            for d in 0..d_n { self.domains[d].read_lods(); }
            for d in 0..d_n { // Every domain...
                let (x, y, z) = get_coordinates_sl(d as u64, self.config.d_x, self.config.d_y); // Own domain coordinate
                let mut offset = self.domains[d].n_lod_own; // buffer write offset
                for dc in 0..d_n { // ...loops over every other domain and writes the extracted LOD to its own LOD buffer
                    if d != dc {
                        let (dx, dy, dz) = get_coordinates_sl(dc as u64, self.config.d_x, self.config.d_y);
                        let dist: i32 = cmp::max((z as i32 - dz as i32).abs(), cmp::max((y as i32 - dy as i32).abs(), (x as i32 - dx as i32).abs()));
                        let depth = cmp::max(0, self.config.mhd_lod_depth as i32 - dist);
                        // This is the range of relevant LOD data for the current foreign domain
                        let range_s = get_offset(depth - 1, dim); // Range start
                        let range_e = get_offset(depth, dim); // Range end
                        // Write to device
                        self.domains[d].qu_lod.as_ref().expect("msg")
                            .write(&self.domains[dc].transfer_lod_host.as_ref().expect("msg")[range_s*4..range_e*4])
                            .offset(offset*4).enq().unwrap();
                        offset += range_e - range_s;
                    }
                }
            }
        }
    }


    // Helper functions
    /// Increments time steps variable for every `LbmDomain`
    fn increment_timestep(&mut self, steps: u32) {
        for d in 0..self.domains.len() {
            self.domains[d].t += steps as u64;
        }
    }

    /// Resets timestep variable for every `LbmDomain`
    fn reset_timestep(&mut self) {
        for d in 0..self.domains.len() {
            self.domains[d].t = 0;
        }
    }

    /// Returns the amount of domains for this lbm
    pub fn get_d_n(&self) -> usize {
        self.domains.len()
    }

    /// Returns the current simulation timestep
    pub fn get_time_step(&self) -> u64 {
        self.domains[0].t
    }
}
