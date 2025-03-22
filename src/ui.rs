//! # ui
//! 
//! Simple gui interface for the simulation implemented with egui. Only supported in single-mode for now. 

use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;
use std::{fs, sync::mpsc, thread};

use eframe::*;
use egui::{Color32, InnerResponse, Label, Sense, Stroke, TextureOptions, Vec2};

use crate::lbm::GraphicsConfig;
use crate::{SimControlTx, SimState};

struct SimControl {
    //SimControl handles control/displaying of the Simulation over the UI
    paused: bool,
    save_img: bool,
    save_file: bool,
    load_file: bool,
    step: u32,
    clear_images: bool,
    frame_spacing: u32,
    frame_spacing_str: String,
    file_name: String,
    file_name_str: String,
    display_img: Option<egui::TextureHandle>,
    camera_rotation: Vec<f32>,
    camera_zoom: f32,

    graphics_cfg: GraphicsConfig,
    cfg_send: bool,
    cfg_init: bool,

    ctrl_tx: mpsc::Sender<SimControlTx>,
    sim_rx: mpsc::Receiver<SimState>,
}

impl SimControl {
    pub fn reset_sim(&mut self) {
        self.paused = true;
    }

    pub fn send_control(&mut self) {
        let tx = if self.cfg_send && self.cfg_init {
            self.cfg_send = false;
            SimControlTx {
                paused: self.paused,
                save_img: self.save_img,
                save_file: self.save_file,
                load_file: self.load_file,
                clear_images: self.clear_images,
                frame_spacing: self.frame_spacing,
                file_name: self.file_name.clone(),
                active: true,
                camera_rotation: self.camera_rotation.clone(),
                camera_zoom: self.camera_zoom,
                graphics_cfg: Some(self.graphics_cfg.clone()),
            }
        } else {
            SimControlTx {
                paused: self.paused,
                save_img: self.save_img,
                save_file: self.save_file,
                load_file: self.load_file,
                clear_images: self.clear_images,
                frame_spacing: self.frame_spacing,
                file_name: self.file_name.clone(),
                active: true,
                camera_rotation: self.camera_rotation.clone(),
                camera_zoom: self.camera_zoom,
                graphics_cfg: None,
            }
        };
        self.ctrl_tx
            .send(tx)
            .expect("GUI cannot communicate with sim thread");

        // save and load commands should only be sent once
        self.save_file = false;
        self.load_file = false;
    }

    // Egui response ro
    fn draw_top_controls(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
        send_control: &mut bool,
    ) -> InnerResponse<InnerResponse<()>> {
        //egui styles
        let zeromargin = egui::Margin {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
        };
        let frame = egui::Frame {
            fill: Color32::WHITE,
            inner_margin: zeromargin,
            outer_margin: zeromargin,
            ..Default::default()
        };
        let transparent_stroke = Stroke {
            width: 1.0,
            color: Color32::TRANSPARENT,
        };
        let blue_stroke = Stroke {
            width: 1.0,
            color: Color32::from_rgb(0xCC, 0xE8, 0xFF),
        };

        egui::TopBottomPanel::top("top_controls")
            .frame(frame)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let spacing = egui::style::Spacing {
                        button_padding: egui::Vec2 { x: 0., y: 0. },
                        ..Default::default()
                    };
                    ui.style_mut().spacing = spacing;
                    ui.style_mut().visuals.widgets.hovered.expansion = 0.0;
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(0xE5, 0xF3, 0xFF);
                    ui.style_mut().visuals.widgets.hovered.bg_stroke = blue_stroke;
                    ui.style_mut().visuals.widgets.inactive.bg_stroke = transparent_stroke;
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::WHITE;
                    if ui
                        .add_sized(
                            [40., 18.],
                            egui::Button::new(if self.paused { "Play" } else { "Pause" }),
                        )
                        .clicked()
                    {
                        self.paused = !self.paused;
                        *send_control = true;
                    }
                    if ui.add(egui::Button::new("Reset")).clicked() {
                        self.reset_sim();
                    }
                    if ui
                        .add(egui::Button::new(if self.save_img {
                            "Saving Enabled"
                        } else {
                            "Saving Disabled"
                        }))
                        .clicked()
                    {
                        self.save_img = !self.save_img;
                        *send_control = true;
                    }
                    ui.add(egui::Label::new("Saving interval:"));
                    let mut text = self.frame_spacing_str.clone();
                    if ui
                        .add(egui::TextEdit::singleline(&mut text).desired_width(50.0))
                        .changed()
                    {
                        self.frame_spacing_str = text.clone(); // clone jik text is cbv here
                        let result = str::parse::<u32>(&text);
                        if let Ok(value) = result {
                            if value > 0 {
                                self.frame_spacing = value;
                                *send_control = true;
                            }
                        }
                    }
                    if ui.add(egui::Button::new("Clear Output")).clicked() {
                        match fs::remove_dir_all("out") {
                            Ok(_) => (),
                            Err(_) => println!("Did not find out folder. Creating it."),
                        }
                        fs::create_dir("out").unwrap();
                    }
                    if ui.add(egui::Button::new("Save Simulation")).clicked() {
                        // do not try to save with no name
                        if !self.file_name.is_empty() {
                            if !self.file_name.ends_with(".ion") {
                                self.file_name.push_str(".ion");
                            }
                            self.save_file = true;
                            *send_control = true;
                        }
                    }
                    if ui.add(egui::Button::new("Load Simulation")).clicked() {
                        // do not try to load with no name
                        if !self.file_name.is_empty() {
                            if !self.file_name.ends_with(".ion") {
                                self.file_name.push_str(".ion");
                            }
                            self.load_file = true;
                            *send_control = true;
                        }
                    }
                    text = self.file_name_str.clone();
                    ui.add(egui::Label::new("File Name:"));
                    if ui
                        .add(egui::TextEdit::singleline(&mut text).desired_width(300.0))
                        .changed()
                    {
                        self.file_name_str = text.clone();

                        let mut temp: String = "./".to_string();
                        temp.push_str(self.file_name_str.clone().as_str());
                        self.file_name = temp;
                    }
                })
            })
    }

    fn draw_graphics_control(&mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) -> InnerResponse<Option<()>> {
        egui::Window::new("Graphics Control")
            .open(&mut true)
            .auto_sized()
            .show(ctx, |ui| {
                egui::Grid::new("my_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(false)
                .show(ui, |ui| {
                    ui.label("Streamlines");
                    if ui.checkbox(&mut self.graphics_cfg.streamline_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Vector field");
                    if ui.checkbox(&mut self.graphics_cfg.field_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Field slice");
                    if ui.checkbox(&mut self.graphics_cfg.field_slice_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();

                    ui.label("Vector Type");
                    egui::ComboBox::from_label(" Type")
                        .selected_text(format!("{}", self.graphics_cfg.vec_vis_mode))
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut self.graphics_cfg.vec_vis_mode, crate::lbm::graphics::VecVisMode::U, "Velocity").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.vec_vis_mode, crate::lbm::graphics::VecVisMode::EStat, "Static E field").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.vec_vis_mode, crate::lbm::graphics::VecVisMode::BStat, "Static B field").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.vec_vis_mode, crate::lbm::graphics::VecVisMode::EDyn, "Dynamic E field").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.vec_vis_mode, crate::lbm::graphics::VecVisMode::BDyn, "Dynamic B field").changed() {self.cfg_send = true;}
                        });
                    ui.end_row();

                    ui.label("Field Mode");
                    egui::ComboBox::from_label(" F Type")
                        .selected_text(format!("{}", self.graphics_cfg.field_vis))
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut self.graphics_cfg.field_vis, crate::lbm::graphics::FieldVisMode::Scalar, "Scalar field").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.field_vis, crate::lbm::graphics::FieldVisMode::Vector, "Vector field").changed() {self.cfg_send = true;}
                        });
                    ui.end_row();

                    ui.label("Slice Mode");
                    egui::ComboBox::from_label(" Mode")
                        .selected_text(format!("{}", self.graphics_cfg.slice_mode))
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::Off, "Off").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::X, "X").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::Y, "Y").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::Z, "Z").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::XZ, "X + Z").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::XYZ, "X + Y + Z").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::YZ, "Y + Z").changed() {self.cfg_send = true;}
                            if ui.selectable_value(&mut self.graphics_cfg.slice_mode, crate::lbm::graphics::SliceMode::XY, "X + Y").changed() {self.cfg_send = true;}
                        });
                    ui.end_row();

                    ui.label("Slice X");
                    if ui.add(egui::Slider::new(&mut self.graphics_cfg.slice_x, 0..=self.graphics_cfg.max_slice_x).suffix(" cells")).changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Slice Y");
                    if ui.add(egui::Slider::new(&mut self.graphics_cfg.slice_y, 0..=self.graphics_cfg.max_slice_y).suffix(" cells")).changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Slice Z");
                    if ui.add(egui::Slider::new(&mut self.graphics_cfg.slice_z, 0..=self.graphics_cfg.max_slice_z).suffix(" cells")).changed() {self.cfg_send = true;}
                    ui.end_row();

                    ui.label("Vorticity");
                    if ui.checkbox(&mut self.graphics_cfg.q_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Vorticity field");
                    if ui.checkbox(&mut self.graphics_cfg.q_field_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Flags");
                    if ui.checkbox(&mut self.graphics_cfg.flags_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Flags Surface");
                    if ui.checkbox(&mut self.graphics_cfg.flags_surface_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("Axes");
                    if ui.checkbox(&mut self.graphics_cfg.axes_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                    ui.label("ECR Condition");
                    if ui.checkbox(&mut self.graphics_cfg.ecrc_mode, "").changed() {self.cfg_send = true;}
                    ui.end_row();
                });
            }).expect("msg")
    }
}

impl App for SimControl {
    /// UI update loop
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        for recieve in self.sim_rx.try_iter() {
            match recieve.img {
                Some(img) => {
                    if self.display_img.is_none() {
                        self.display_img =
                            Some(ctx.load_texture("sim", img.clone(), Default::default()))
                    }
                    if recieve.step >= self.step {
                        self.display_img
                            .as_mut()
                            .expect("Isn't TextureHandle")
                            .set(img, TextureOptions::default());
                    }
                },
                None => {},
            }
            
            match recieve.graphics_cfg {
                Some(cfg) => {self.graphics_cfg = cfg; self.cfg_init = true;},
                None => {},
            }
            if self.step < recieve.step {
                self.step = recieve.step;
            }
        }

        let mut send_control = false; // determines if ui needs to send control

        //egui styles
        let zeromargin = egui::Margin {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
        };
        let small_left_margin = egui::Margin {
            left: 0,
            right: 0,
            top: 1,
            bottom: -1,
        };
        let mut frame = egui::Frame {
            fill: Color32::WHITE,
            inner_margin: small_left_margin,
            outer_margin: zeromargin,
            ..Default::default()
        };

        //Update gui
        self.draw_top_controls(ctx, _frame, &mut send_control);

        frame.outer_margin = zeromargin;
        frame.fill = Color32::from_rgb(40, 40, 40);
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            ui.style_mut().visuals.override_text_color = Some(Color32::WHITE);
            ui.vertical_centered_justified(|ui| {
                match &self.display_img {
                    Some(img) => {
                        let response = ui.add(
                            egui::Image::new(img)
                                .fit_to_fraction(egui::vec2(1.0, 1.0))
                        );
                        let id = ui.id();
                        let response = ui.interact(response.rect, id, Sense::click_and_drag());
                        if response.dragged() {
                            let delta = response.drag_delta();
                            if delta != Vec2::ZERO {
                                self.camera_rotation[0] += delta.x;
                                self.camera_rotation[1] =
                                    (self.camera_rotation[1] - delta.y).clamp(-90.0, 90.0);
                                send_control = true;
                            }
                        }
                    }
                    None => {
                        ui.style_mut()
                            .text_styles
                            .get_mut(&egui::TextStyle::Body)
                            .unwrap()
                            .size = 18.0;
                        ui.with_layout(
                            egui::Layout::centered_and_justified(egui::Direction::TopDown),
                            |ui| {
                                ui.label("Simulation Graphic Output");
                                ui.add(Label::new("text"));
                            },
                        );
                    }
                }
            });
            
            ui.input(|i| {
                if i.raw_scroll_delta != Vec2::ZERO {
                    self.camera_zoom = (self.camera_zoom
                        + (self.camera_zoom * (i.raw_scroll_delta.y * 0.001)))
                        .max(0.01);
                    send_control = true;
                }
            });
        });

        self.draw_graphics_control(ctx, _frame);

        if send_control || self.cfg_send {
            self.send_control()
        }

        if !self.paused {
            ctx.request_repaint_after(Duration::from_millis(100));
        } else {
            ctx.request_repaint_after(Duration::from_millis(1000));
        }

        thread::spawn(move || {});
    }
}

pub fn load_icon(icon_bytes: &[u8]) -> Option<Arc<egui::IconData>> {
    if let Ok(image) = image::load_from_memory(icon_bytes) {
        let image = image.to_rgba8();
        let (width, height) = image.dimensions();
        Some(Arc::new(egui::IconData {
            width,
            height,
            rgba: image.as_raw().to_vec(),
        }))
    } else {
        None
    }
}

pub fn run_ui(sim_rx: Receiver<SimState>, ctrl_tx: Sender<SimControlTx>) {
    let icon_bytes = include_bytes!("../icons/IonSolver.png");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder {
            title: Some("IonSolver".into()),
            maximized: Some(true),
            icon: load_icon(icon_bytes),
            ..Default::default()
        },
        //initial_window_size: Some(egui::vec2(1000.0, 650.0)),
        //follow_system_theme: false,
        //default_theme: Theme::Light,
        //icon_data: load_icon(icon_bytes.as_ref()),
        ..Default::default()
    };

    // setup simcontrol struct to pass params and channels to GUI
    let simcontrol = SimControl {
        paused: true,
        save_img: false,
        save_file: false,
        load_file: false,
        step: 0,
        clear_images: false,
        frame_spacing: 100,
        frame_spacing_str: "100".to_string(),
        file_name: "".to_string(),
        file_name_str: "".to_string(),
        display_img: None,
        camera_rotation: vec![0.0; 2],
        camera_zoom: 3.0,
        ctrl_tx,
        sim_rx,
        graphics_cfg: GraphicsConfig::new(),
        cfg_init: false,
        cfg_send: false,
    };

    // Start window loop
    eframe::run_native(
        "IonSolver",
        options,
        Box::new(|_| Ok(Box::<SimControl>::new(simcontrol))),
    )
    .expect("unable to open window");
}
