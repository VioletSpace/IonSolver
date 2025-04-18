//! # setup
//! 
//! Holds functions for various simulation setups. 
//! 
//! `setup()` is called in single-node mode to create a new Lbm struct. Modify this function to create your own setups or configure it to read a setup from file.

use crate::*;

#[allow(unused)]
/// `setup()` is called at simulation start. Edit this function to change simulation parameters.
///
/// Usage:
///
/// Start by requesting a new `LbmConfig` with `LbmConfig::new()`.
/// You can then set individual arguments by setting fields of the `LbmConfig`.
/// If you are doing something that requires real-world units/scales, you can define them using `your_lbm_config.units.set()` and pass in your desired units.
///
/// After setting your config struct, request a new `Lbm` struct with `Lbm::new(your_lbm_config)`.
/// Domain setup is handled automatically, you might now directly set specific cells in your domains, for an example look at `setup_taylor_green()`.
///
/// Run `your_lbm.initialize()` and return it with the config.
pub fn setup() -> Lbm {
    
    //let mut cfg = LbmConfig::new();
    //cfg.n_x = 128;
    //cfg.n_y = 256;
    //cfg.n_z = 128;
    //cfg.nu = cfg.units.nu_si_lu(0.05);
    //cfg.velocity_set = VelocitySet::D3Q19;
    // Graphics
    //cfg.graphics_config.graphics_active = true;
    //cfg.graphics_config.streamline_every = 8;
    //cfg.graphics_config.vec_vis_mode = graphics::VecVisMode::U;
    //cfg.graphics_config.streamline_mode = true;
    //cfg.graphics_config.axes_mode = true;
    //cfg.graphics_config.q_mode = true;
    //cfg.graphics_config.q_min = 0.00001;
    //let mut lbm = Lbm::new(cfg);
    //lbm.domains[0].q.as_ref().expect("msg").write(&charge).enq().unwrap();
    //lbm.setup_velocity_field((0.01, 0.001, 0.0), 1.0);
    //lbm.set_taylor_green(1);
    //lbm.import_mesh("stl/cow.stl", 0.0, 64.0, 128.0, 64.0, -30.0, 0.0, 0.0);

    //lbm
    
    
    //setup_domain_test()
    //setup_bfield_spin()
    //file::write(&setup_bfield_spin(), "./testfile.ion");
    //setup_from_file("./testfile.ion")
    //let mut lbm = setup_taylor_green();
    //setup_verification()
    //setup_field_vis()
    //setup_ecr_test()
    //setup_mesh_test()
    //setup_mesh_field_test()
    setup_deeva_test()

    //let now = std::time::Instant::now();
    //lbm.run(10);
    //println!("10 time steps with 416^3 cells took: {}", now.elapsed().as_millis());
    //lbm
    
}

#[allow(unused)]
/// Set up Lbm from file.
/// Use in setup();
///
/// Format documented in FILE_LAYOUT.txt
pub fn setup_from_file(path: &str) -> Lbm {
    let mut config = LbmConfig::new();

    // Graphics setup because these are not saved
    config.graphics_config.graphics_active = true;
    config.graphics_config.background_color = 0x1c1b22;
    config.graphics_config.camera_width = 1920;
    config.graphics_config.camera_height = 1080;
    config.graphics_config.streamline_every = 8;
    config.graphics_config.vec_vis_mode = graphics::VecVisMode::U;
    config.graphics_config.streamline_mode = true;
    config.graphics_config.axes_mode = true;
    config.graphics_config.q_mode = false;
    config.graphics_config.flags_surface_mode = true;
    config.graphics_config.flags_mode = true;

    file::read(path, &mut config).unwrap()
}

// Complete setups
#[allow(unused)]
fn setup_taylor_green() -> Lbm {
    let mut lbm_config = LbmConfig::new();
    lbm_config.n_x = 428;
    lbm_config.n_y = 428;
    lbm_config.n_z = 428;
    lbm_config.nu = lbm_config.units.nu_si_lu(0.1);
    lbm_config.velocity_set = VelocitySet::D3Q19;
    // Graphics
    lbm_config.graphics_config.graphics_active = true;
    lbm_config.graphics_config.streamline_every = 8;
    lbm_config.graphics_config.vec_vis_mode = graphics::VecVisMode::U;
    //lbm_config.graphics_config.streamline_mode = true;
    lbm_config.graphics_config.axes_mode = true;
    lbm_config.graphics_config.q_mode = true;
    lbm_config.graphics_config.q_min = 0.00001;

    let mut lbm = Lbm::new(lbm_config);
    lbm.set_taylor_green(1);

    lbm
}

#[allow(unused)]
fn setup_domain_test() -> Lbm {
    let mut lbm_config = LbmConfig::new();
    lbm_config.units.set(128.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.2250, 1.0, 1.0);
    lbm_config.n_x = 256;
    lbm_config.n_y = 256;
    lbm_config.n_z = 256;
    lbm_config.d_z = 2; // Two domains on z-axis (128 cells long each)
    lbm_config.velocity_set = VelocitySet::D3Q19;

    lbm_config.nu = lbm_config.units.nu_si_lu(0.00148);

    // Graphics
    lbm_config.graphics_config.graphics_active = true;
    lbm_config.graphics_config.streamline_every = 8;
    lbm_config.graphics_config.vec_vis_mode = graphics::VecVisMode::U;
    lbm_config.graphics_config.streamline_mode = true;
    lbm_config.graphics_config.field_mode = false;
    lbm_config.graphics_config.u_max = 0.3;
    lbm_config.graphics_config.axes_mode = true;

    let mut lbm = Lbm::new(lbm_config.clone());
    lbm.set_taylor_green(1);

    lbm
}

#[allow(unused)]
fn setup_bfield_spin() -> Lbm {
    let mut lbm_config = LbmConfig::new();
    lbm_config.units.set(128.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.2250, 0.0000000001, 1.0);
    lbm_config.units.print();
    lbm_config.n_x = 128;
    lbm_config.n_y = 128;
    lbm_config.n_z = 256;
    lbm_config.d_z = 2;
    lbm_config.nu = lbm_config.units.nu_si_lu(1.48E-5);
    println!("    nu in LU is: {}", lbm_config.units.nu_si_lu(1.48E-3));
    lbm_config.velocity_set = VelocitySet::D3Q19;
    lbm_config.mhd_lod_depth = 4;
    // Extensions
    lbm_config.ext_volume_force = true;
    lbm_config.ext_magneto_hydro = true;
    // Graphics
    lbm_config.graphics_config.graphics_active = true;
    lbm_config.graphics_config.background_color = 0x1c1b22;
    lbm_config.graphics_config.camera_width = 1920;
    lbm_config.graphics_config.camera_height = 1080;
    lbm_config.graphics_config.streamline_every = 8;
    lbm_config.graphics_config.vec_vis_mode = graphics::VecVisMode::EDyn;
    lbm_config.graphics_config.u_max = 500.0;
    lbm_config.graphics_config.streamline_mode = true;
    lbm_config.graphics_config.axes_mode = true;
    lbm_config.graphics_config.q_mode = false;
    lbm_config.graphics_config.flags_surface_mode = true;
    lbm_config.graphics_config.flags_mode = true;

    let mut lbm = Lbm::new(lbm_config);

    let mut flags: Vec<u8> = vec![0; 128 * 128 * 128];
    let len = flags.len() - 1;
    // Solid
    for i in 0..(128 * 128 * 8) {
        flags[i] = 0x01;
        //flags[len - i] = 0x01;
    }
    bwrite!(lbm.domains[0].flags, flags);

    let cpc = 0.002;
    let mut charge: Vec<f32> = vec![cpc; 128 * 128 * 128];
    bwrite!(lbm.domains[0].q.as_ref().expect("q"), charge);

    // magnetic field
    let mut vec_m: Vec<(u64, [f32; 3])> = vec![];
    for i in 0..243 {
        vec_m.push((i * 68, [0.0, 0.0, 100000000000000000.0]));
        vec_m.push((i * 68 + 2097152 * 2, [0.0, 0.0, 100000000000000000.0]));
    }

    //lbm.magnets = Some(vec_m);
    //precompute::precompute_B(&lbm);
    //precompute::precompute_E(&lbm);

    lbm.setup_velocity_field((0.1, 0.01, 0.0), 1.0);

    lbm
}

#[allow(unused)]
fn setup_verification() -> Lbm {
    let mut lbm_config = LbmConfig::new();
    lbm_config.units.set(1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 10.0, 1.0);
    lbm_config.units.print();
    lbm_config.n_x = 128;
    lbm_config.n_y = 128;
    lbm_config.n_z = 128;
    lbm_config.nu = lbm_config.units.nu_si_lu(1.48E-5);
    println!("    nu in LU is: {}", lbm_config.units.nu_si_lu(1.48E-3));
    lbm_config.velocity_set = VelocitySet::D3Q19;
    lbm_config.mhd_lod_depth = 4;
    // Extensions
    lbm_config.ext_volume_force = true;
    lbm_config.ext_magneto_hydro = true;
    // Graphics
    lbm_config.graphics_config.graphics_active = true;
    lbm_config.graphics_config.background_color = 0x1c1b22;
    lbm_config.graphics_config.camera_width = 1920;
    lbm_config.graphics_config.camera_height = 1080;
    lbm_config.graphics_config.streamline_every = 8;
    lbm_config.graphics_config.vec_vis_mode = graphics::VecVisMode::BStat;
    lbm_config.graphics_config.streamline_mode = true;
    lbm_config.graphics_config.axes_mode = true;
    lbm_config.graphics_config.q_mode = true;
    lbm_config.graphics_config.flags_surface_mode = true;
    lbm_config.graphics_config.flags_mode = true;

    let lbm = Lbm::new(lbm_config);

    let mut charge = vec![0.0; (lbm.config.n_x * lbm.config.n_y * lbm.config.n_z) as usize];
    charge[0] = 1.0; // 1C
    bwrite!(lbm.domains[0].q.as_ref().expect("q"), charge);
    let mut vel = vec![0.0; (lbm.config.n_x * lbm.config.n_y * lbm.config.n_z * 3) as usize];
    vel[0] = 1.0; // 1m/s in x+
    bwrite!(lbm.domains[0].u, vel);
    let rho = vec![1.0; (lbm.config.n_x * lbm.config.n_y * lbm.config.n_z) as usize];
    bwrite!(lbm.domains[0].rho, rho);
    lbm
}

#[allow(unused)]
fn setup_field_vis() -> Lbm {
    let mut lbm_config = LbmConfig::new();
    lbm_config.units.set(128.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.2250, 0.0000000001, 1.0);
    lbm_config.n_x = 256;
    lbm_config.n_y = 256;
    lbm_config.n_z = 256;
    lbm_config.nu = lbm_config.units.nu_si_lu(1.48E-5);
    lbm_config.velocity_set = VelocitySet::D3Q19;
    lbm_config.mhd_lod_depth = 4;
    // Extensions
    lbm_config.ext_volume_force = true;
    lbm_config.ext_magneto_hydro = true;
    // Graphics
    lbm_config.graphics_config.graphics_active = true;
    lbm_config.graphics_config.background_color = 0x000000;
    lbm_config.graphics_config.camera_width = 800;
    lbm_config.graphics_config.camera_height = 540;
    lbm_config.graphics_config.streamline_every = 16;
    lbm_config.graphics_config.vec_vis_mode = graphics::VecVisMode::BStat;
    lbm_config.graphics_config.u_max = 0.000000002;
    lbm_config.graphics_config.streamline_mode = true;

    let mut lbm = Lbm::new(lbm_config);

    // magnetic field
    let mut vec_m: Vec<(u64, [f32; 3])> = vec![];
    for i in 0..40 {
        vec_m.push((127 - 20 + i + 32768 + 8388608, [1000000000.0f32, 0.0f32, 0.0f32]));
    }
    //lbm.magnets = Some(vec_m);
    //precompute::precompute_B(&lbm);

    lbm
}

#[allow(unused)]
fn setup_ecr_test() -> Lbm {
    let mut lbm_config = LbmConfig::new();
    lbm_config.n_x = 1;
    lbm_config.n_y = 1;
    lbm_config.n_z = 1;
    lbm_config.velocity_set = VelocitySet::D3Q19;
    lbm_config.ext_volume_force = true;
    lbm_config.ext_magneto_hydro = true;
    lbm_config.ext_subgrid_ecr = true;
    lbm_config.ecr_freq = 0.1f32;
    lbm_config.ecr_field_strength = 1.0f32;

    let mut lbm = Lbm::new(lbm_config);

    lbm
}

#[allow(unused)]
fn setup_mesh_test() -> Lbm {
    let mut cfg = LbmConfig::new();
    cfg.n_x = 128;
    cfg.n_y = 256;
    cfg.n_z = 128;
    cfg.nu = cfg.units.nu_si_lu(0.05);
    cfg.velocity_set = VelocitySet::D3Q19;
    // Graphics
    cfg.graphics_config.graphics_active = true;
    cfg.graphics_config.flags_surface_mode = true;
    cfg.graphics_config.axes_mode = true;
    let mut lbm = Lbm::new(cfg);

    // Cow mesh
    lbm.import_mesh_reposition("stl/cow.stl", 64.0, 128.0, 64.0, 0.0, 0.0, 0.0, 200.0);
    let off = 128.0-lbm.meshes[0].p_max.z;
    lbm.meshes[0].translate(mesh::v3(0.0, 0.0, off));
    lbm.voxelise_mesh(0, mesh::ModelType::Solid);

    lbm.import_mesh_reposition("stl/ring-magnet.stl", 64.5, 10.0, 64.0, 0.0, 0.0, 0.0, 127.0);
    lbm.voxelise_mesh(1, mesh::ModelType::Magnet {magnetization: (0.0, 0.0, 0.0)});

    lbm
}

#[allow(unused)]
fn setup_mesh_field_test() -> Lbm {
    let mut cfg = LbmConfig::new();
    cfg.units.set(128.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.2250, 0.0000000001, 1.0);
    cfg.n_x = 128;
    cfg.n_y = 256;
    cfg.n_z = 128;
    cfg.nu = cfg.units.nu_si_lu(0.05);
    cfg.velocity_set = VelocitySet::D3Q19;
    cfg.ecr_freq = cfg.units.time_lu_si(2.45E9); // 2.45 gigahertz
    // Extensions
    cfg.ext_volume_force = true;
    cfg.ext_magneto_hydro = true;
    // Graphics
    cfg.graphics_config.graphics_active = true;
    cfg.graphics_config.streamline_every = 8;
    cfg.graphics_config.vec_vis_mode = graphics::VecVisMode::BStat;
    cfg.graphics_config.u_max = 0.00002;
    cfg.graphics_config.streamline_mode = true;
    cfg.graphics_config.streamline_every = 16;
    cfg.graphics_config.axes_mode = true;
    cfg.graphics_config.flags_surface_mode = true;
    cfg.graphics_config.flags_mode = false;
    cfg.graphics_config.ecrc_mode = true;
    let mut lbm = Lbm::new(cfg);

    // Ring magnet
    //lbm.import_mesh_reposition("stl/ring-magnet.stl", 64.5, 128.5, 64.0, 0.0, 0.0, 0.0, 100.0);
    //lbm.voxelise_mesh(0, mesh::ModelType::Magnet {magnetization: (0.0, 1200000.0, 0.0)});

    // Magnetic cow
    //lbm.import_mesh_reposition("stl/cow.stl", 64.0, 128.0, 64.0, 0.0, 0.0, 0.0, 166.0);
    //let off = 128.0-lbm.meshes[0].p_max.z;
    //lbm.meshes[0].translate(mesh::v3(0.0, 0.0, off));
    //lbm.voxelise_mesh(0, mesh::ModelType::Magnet {magnetization: (0.0, 10000.0, 0.0)});

    // Thruster geometry
    //lbm.import_mesh_reposition("stl/ring-magnet.stl", 64.1, 64.1, 64.0, 0.0, 0.0, 0.0, 127.0);
    lbm.import_mesh_reposition("stl/disk-magnet.stl", 64.1, 246.1, 64.0, 0.0, 0.0, 0.0, 127.0);
    lbm.import_mesh("stl/ring-magnet.stl", 1.0, 64.1, 64.1, 64.0, 0.0, 0.0, 0.0);
    lbm.voxelise_mesh(0, mesh::ModelType::Magnet {magnetization: (0.0, 1000000.0, 0.0)});
    lbm.voxelise_mesh(1, mesh::ModelType::Magnet {magnetization: (0.0, 1000000.0, 0.0)});
    //lbm.precompute_B();

    lbm.domains[0].dump_cell(((200 + 64 * lbm.config.n_y) * lbm.config.n_x) as usize, &lbm.config);

    lbm
}

fn setup_deeva_test() -> Lbm {
    let mut cfg = LbmConfig::new();
    cfg.units.set(128.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.2250, 0.0000000001, 1.0);
    cfg.n_x = 128;
    cfg.n_y = 256;
    cfg.n_z = 128;
    cfg.nu = cfg.units.nu_si_lu(0.05);
    cfg.velocity_set = VelocitySet::D3Q19;
    cfg.ecr_freq = cfg.units.time_lu_si(2.45E9); // 2.45 gigahertz
    // Extensions
    cfg.ext_volume_force = true;
    cfg.ext_magneto_hydro = true;
    // Graphics
    cfg.graphics_config.graphics_active = true;
    cfg.graphics_config.streamline_every = 8;
    cfg.graphics_config.vec_vis_mode = graphics::VecVisMode::BStat;
    cfg.graphics_config.u_max = 0.00003;
    cfg.graphics_config.streamline_mode = true;
    cfg.graphics_config.streamline_every = 16;
    cfg.graphics_config.axes_mode = true;
    cfg.graphics_config.flags_surface_mode = true;
    cfg.graphics_config.flags_mode = false;
    cfg.graphics_config.ecrc_mode = true;
    let mut lbm = Lbm::new(cfg);
    

    // Thruster geometry
    //lbm.import_mesh_reposition("stl/ring-magnet.stl", 64.1, 64.1, 64.0, 0.0, 0.0, 0.0, 127.0);
    //lbm.import_mesh_reposition("stl/disk-magnet.stl", 64.1, 246.1, 64.0, 0.0, 0.0, 0.0, 127.0);
    lbm.import_mesh("stl/deeva_disk_magnet.stl", 1.0, 64.001, 0.0, 64.0, 0.0, 0.0, 0.0);
    lbm.import_mesh("stl/deeva_inlet.stl", 1.0, 64.0, 0.0, 64.0, 0.0, 0.0, 0.0);
    lbm.import_mesh("stl/deeva_quartz_tube.stl", 1.0, 64.001, 0.0, 64.0, 0.0, 0.0, 0.0);
    lbm.import_mesh("stl/deeva_ring_magnet.stl", 1.0, 64.001, -0.5, 64.0, 0.0, 0.0, 0.0);
    lbm.voxelise_mesh(0, mesh::ModelType::Magnet {magnetization: (0.0, 1000000.0, 0.0)});
    lbm.voxelise_mesh(1, mesh::ModelType::Solid);
    lbm.voxelise_mesh(2, mesh::ModelType::Solid);
    lbm.voxelise_mesh(3, mesh::ModelType::Magnet {magnetization: (0.0, 500000.0, 0.0)});
    lbm.precompute_B();

    lbm.domains[0].dump_cell((64 + (50 + 64 * lbm.config.n_y) * lbm.config.n_x) as usize, &lbm.config);

    lbm
}

impl Lbm {
    #[allow(unused)]
    /// 3D Taylor-Green vorticies setup.
    pub fn set_taylor_green(&mut self, periodicity: u32) {
        println!("Setting up Taylor-Green vorticies");
        let nx = self.config.n_x;
        let ny = self.config.n_y;
        let nz = self.config.n_z;

        let pif = std::f32::consts::PI;
        #[allow(non_snake_case)]
        let A = 0.25f32;
        let a = nx as f32 / periodicity as f32;
        let b = ny as f32 / periodicity as f32;
        let c = nz as f32 / periodicity as f32;

        let domain_numbers: u32 = self.config.d_x * self.config.d_y * self.config.d_z;
        let dx = self.config.d_x;
        let dy = self.config.d_y;
        let dz = self.config.d_z;
        let dsx = nx as u64 / dx as u64 + (dx > 1u32) as u64 * 2; // Domain size on each axis
        let dsy = ny as u64 / dy as u64 + (dy > 1u32) as u64 * 2; // Needs to account for halo offsets
        let dsz = nz as u64 / dz as u64 + (dz > 1u32) as u64 * 2;
        let dtotal = dsx * dsy * dsz;

        print!("");
        for d in 0..domain_numbers {
            let x = (d % (dx * dy)) % dx; // Current Domain coordinates
            let y = (d % (dx * dy)) / dx;
            let z = d / (dx * dy);

            let mut domain_vec_u: Vec<f32> = vec![0.0; (dsx * dsy * dsz * 3) as usize];
            let mut domain_vec_rho: Vec<f32> = vec![0.0; (dsx * dsy * dsz) as usize];
            for zi in 0..dsz {
                // iterates over every cell in the domain, filling it with  Taylor-Green-vortex
                print!(
                    "\r{}",
                    info::progressbar(
                        ((zi as f32 + 1.0) / dsz as f32) * (1.0 / domain_numbers as f32)
                            + ((d as f32) / domain_numbers as f32)
                    )
                );
                for yi in 0..dsy {
                    for xi in 0..dsx {
                        if !(((xi == 0 || xi == dsx - 1) && dx > 1)
                            || ((yi == 0 || yi == dsy - 1) && dy > 1)
                            || ((zi == 0 || zi == dsz - 1) && dz > 1))
                        {
                            //do not set at halo offsets
                            let dn = (zi * dsx * dsy) + (yi * dsx) + xi; // Domain 1D index
                            let gx =
                                xi - (dx > 1u32) as u64 + x as u64 * (dsx - (dx > 1u32) as u64 * 2); // Global coordinates
                            let gy =
                                yi - (dy > 1u32) as u64 + y as u64 * (dsy - (dy > 1u32) as u64 * 2);
                            let gz =
                                zi - (dz > 1u32) as u64 + z as u64 * (dsz - (dz > 1u32) as u64 * 2);

                            let fx = gx as f32 + 0.5 - 0.5 * nx as f32;
                            let fy = gy as f32 + 0.5 - 0.5 * ny as f32;
                            let fz = gz as f32 + 0.5 - 0.5 * nz as f32;

                            domain_vec_u[(dn) as usize] = A
                                * (2.0 * pif * fx / a).cos()
                                * (2.0 * pif * fy / b).sin()
                                * (2.0 * pif * fz / c).sin(); // x
                            domain_vec_u[(dn + dtotal) as usize] = -A
                                * (2.0 * pif * fx / a).sin()
                                * (2.0 * pif * fy / b).cos()
                                * (2.0 * pif * fz / c).sin(); // y;
                            domain_vec_u[(dn + dtotal * 2) as usize] = A
                                * (2.0 * pif * fx / a).sin()
                                * (2.0 * pif * fy / b).sin()
                                * (2.0 * pif * fz / c).cos(); // z
                            domain_vec_rho[(dn) as usize] = 1.0
                                - (A * A) * 3.0 / 4.0 * (4.0 * pif * fx / a).cos()
                                + (4.0 * pif * fy / b).cos();
                        }
                    }
                }
            }

            // Write to domain buffers
            bwrite!(self.domains[d as usize].u, domain_vec_u);
            bwrite!(self.domains[d as usize].rho, domain_vec_rho);
            self.domains[d as usize].queue.finish().unwrap();
        }
        println!();
        println!("Finished setting up Taylor-Green vorticies");
    }

    #[allow(unused)]
    /// Sets all fluid cells to the specified velocity
    fn setup_velocity_field(&mut self, velocity: (f32, f32, f32), density: f32) {
        println!("Setting up velocity field");
        let nx = self.config.n_x;
        let ny = self.config.n_y;
        let nz = self.config.n_z;
        let domain_numbers: u32 = self.config.d_x * self.config.d_y * self.config.d_z;
        let dx = self.config.d_x;
        let dy = self.config.d_y;
        let dz = self.config.d_z;
        let dsx = nx as u64 / dx as u64 + (dx > 1u32) as u64 * 2; // Domain size on each axis
        let dsy = ny as u64 / dy as u64 + (dy > 1u32) as u64 * 2; // Needs to account for halo offsets
        let dsz = nz as u64 / dz as u64 + (dz > 1u32) as u64 * 2;
        let dtotal = dsx * dsy * dsz;

        print!("");
        for d in 0..domain_numbers {
            let mut domain_vec_u: Vec<f32> = vec![0.0; (dsx * dsy * dsz * 3) as usize];
            let mut domain_vec_rho: Vec<f32> = vec![0.0; (dsx * dsy * dsz) as usize];
            for zi in 0..dsz {
                // iterates over every cell in the domain, filling it with the velocity field
                print!(
                    "\r{}",
                    info::progressbar(
                        ((zi as f32 + 1.0) / dsz as f32) * (1.0 / domain_numbers as f32)
                            + ((d as f32) / domain_numbers as f32)
                    )
                );
                for yi in 0..dsy {
                    for xi in 0..dsx {
                        if !(((xi == 0 || xi == dsx - 1) && dx > 1)
                            || ((yi == 0 || yi == dsy - 1) && dy > 1)
                            || ((zi == 0 || zi == dsz - 1) && dz > 1))
                        {
                            //do not set at halo offsets
                            let dn = (zi * dsx * dsy) + (yi * dsx) + xi; // Domain 1D index
                            domain_vec_u[(dn) as usize] = velocity.0; // x
                            domain_vec_u[(dn + dtotal) as usize] = velocity.1; // y;
                            domain_vec_u[(dn + dtotal * 2) as usize] = velocity.2; // z
                            domain_vec_rho[(dn) as usize] = density;
                        }
                    }
                }
            }

            // Write to domain buffers
            bwrite!(self.domains[d as usize].u, domain_vec_u);
            bwrite!(self.domains[d as usize].rho, domain_vec_rho);
            self.domains[d as usize].queue.finish().unwrap();
        }
        println!();
        println!("Finished setting up velocity field");
    }
}
