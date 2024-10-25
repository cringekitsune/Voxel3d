use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    math::vec3,
    pbr::*,
    prelude::*,
    render::{
        mesh::{self, *},
        render_asset::*,
    },
    utils::HashMap,
    window::{PresentMode, WindowTheme},
};
use bevy_flycam::{FlyCam, NoCameraPlayerPlugin};
use noise::{NoiseFn, Perlin};
use rand::Rng;
use std::f32::consts::PI;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Voxel3d".into(),
                        name: Some("cringekitsune.voxel3d.game".into()),
                        resolution: (1080., 720.).into(),
                        present_mode: PresentMode::AutoNoVsync,
                        window_theme: Some(WindowTheme::Dark),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .set(ImagePlugin::default_nearest()),
            NoCameraPlayerPlugin,
            FrameTimeDiagnosticsPlugin,
        ))
        .add_systems(Startup, (setup, setup_fps_counter))
        .add_systems(
            Update,
            (fps_text_update_system, fps_counter_showhide, update_terrain),
        )
        .run();
}

fn print_camera_position(query: Query<&Transform, With<Camera>>) {
    if let Ok(transform) = query.get_single() {
        println!("Camera position: {:?}", transform.translation);
    }
}

fn setup(
    mut commands: Commands,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        },
        FogSettings {
            color: Color::srgba(0.45, 0.58, 0.66, 1.0),
            directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
            directional_light_exponent: 90.0,
            falloff: FogFalloff::from_visibility_colors(
                512.0,
                Color::srgb(0.35, 0.5, 0.66),
                Color::srgb(0.8, 0.844, 1.0),
            ),
        },
        FlyCam,
    ));

    commands.insert_resource(ActiveChunks {
        chunks: HashMap::new(),
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            PI / 4.,
            -PI / 4.,
        )),
        cascade_shadow_config: CascadeShadowConfigBuilder {
            first_cascade_far_bound: 7.0,
            maximum_distance: 512.0,
            ..default()
        }
        .into(),
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: mesh_assets.add(Mesh::from(Cuboid::new(1000.0, 1000.0, 1000.0))), // Make it giant with size 10
            material: material_assets.add(StandardMaterial {
                base_color: Srgba::hex("000000").unwrap().into(),
                unlit: true,
                cull_mode: None,
                ..Default::default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..Default::default()
        },
        NotShadowCaster,
    ));
}

fn dist3d(p1: (f64, f64, f64), p2: (f64, f64, f64)) -> f64 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let dz = p2.2 - p1.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

use std::collections::HashSet;

#[derive(Resource)]
struct ActiveChunks {
    chunks: HashMap<(i32, i32, i32), Entity>,
}


fn update_terrain(
    mut commands: Commands,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    query: Query<&Transform, With<Camera>>,
    mut active_chunks: ResMut<ActiveChunks>,
) {
    if let Ok(transform) = query.get_single() {
        let camera_position = transform.translation;

        let chunk_size = 64;
        let render_distance = 2;

        let center_chunk_x = (camera_position.x / chunk_size as f32).floor() as i32;
        let center_chunk_y = (camera_position.y / chunk_size as f32).floor() as i32;
        let center_chunk_z = (camera_position.z / chunk_size as f32).floor() as i32;

        let custom_texture_handle: Handle<Image> = asset_server.load("textures/texture.png");
        let normal_map: Handle<Image> = asset_server.load("textures/normal.png");
        let cube_material = material_assets.add(StandardMaterial {
            base_color_texture: Some(custom_texture_handle),
            normal_map_texture: Some(normal_map),
            ..Default::default()
        });

        let mut new_chunks = HashSet::new();

        let max_chunks_per_frame = 16;
        let mut chunks_loaded = 0;

        for x in -render_distance..=render_distance {
            for y in -render_distance..=render_distance {
                for z in -render_distance..=render_distance {
                    let chunk_x = center_chunk_x + x;
                    let chunk_y = center_chunk_y + y;
                    let chunk_z = center_chunk_z + z;

                    if !active_chunks
                        .chunks
                        .contains_key(&(chunk_x, chunk_y, chunk_z))
                    {
                        if chunks_loaded < max_chunks_per_frame {
                            let chunk_mesh =
                                generate_chunk(chunk_x, chunk_y, chunk_z, chunk_size, 1);
                            let chunk_entity = commands
                                .spawn(PbrBundle {
                                    mesh: mesh_assets.add(chunk_mesh),
                                    material: cube_material.clone(),
                                    transform: Transform::from_translation(Vec3::new(
                                        chunk_x as f32 * chunk_size as f32,
                                        chunk_y as f32 * chunk_size as f32,
                                        chunk_z as f32 * chunk_size as f32,
                                    )),
                                    ..Default::default()
                                })
                                .id();

                            active_chunks
                                .chunks
                                .insert((chunk_x, chunk_y, chunk_z), chunk_entity);
                            chunks_loaded += 1;
                        } else {
                            break;
                        }
                    }

                    new_chunks.insert((chunk_x, chunk_y, chunk_z));
                }
            }
        }

        let chunks_to_remove: Vec<(i32, i32, i32)> = active_chunks
            .chunks
            .keys()
            .filter(|&pos| !new_chunks.contains(pos))
            .cloned()
            .collect();

        for chunk_pos in chunks_to_remove {
            if let Some(chunk_entity) = active_chunks.chunks.remove(&chunk_pos) {
                commands.entity(chunk_entity).despawn();
            }
        }

        active_chunks
            .chunks
            .retain(|key, _| new_chunks.contains(key));
        for &pos in &new_chunks {
            if !active_chunks.chunks.contains_key(&pos) {
            }
        }
    }
}

fn generate_chunk(chunk_x: i32, chunk_y: i32, chunk_z: i32, chunk_size: usize, lod: usize) -> Mesh {
    let perlin = Perlin::new(192);
    let perlin_variation = Perlin::new(284822842);
    let scale = 0.025;
    let variation_scale = 0.055;

    let mut chunk_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );

    let mut terrain = vec![vec![vec![false; chunk_size]; chunk_size]; chunk_size];

    // Generate terrain data
    for x in (0..chunk_size).step_by(lod) {
        for z in (0..chunk_size).step_by(lod) {
            let variation_value = perlin_variation.get([
                (x as f64 + chunk_x as f64 * chunk_size as f64) * variation_scale, // Adjust scale if needed
                (z as f64 + chunk_z as f64 * chunk_size as f64) * variation_scale,
            ]);
            for y in (0..chunk_size).step_by(lod) {
                let noise_value = perlin.get([
                    (x as f64 + chunk_x as f64 * chunk_size as f64) * scale,
                    (y as f64 + chunk_y as f64 * chunk_size as f64) * scale,
                    (z as f64 + chunk_z as f64 * chunk_size as f64) * scale,
                ]);
                
                // Combine the noise value with the variation
                if noise_value > (0.3 + (0.3 + ((chunk_y as f64 * chunk_size as f64) + y as f64) * (0.07 + (variation_value * 0.06)))) {
                    terrain[x][y][z] = true;
                }
            }
        }
    }

    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut vertex_offset = 0;
    let mut tangents = Vec::new();

    let mut rng = rand::thread_rng();

    // Create the mesh
    for x in 0..chunk_size {
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                if terrain[x][y][z] {
                    let show_top =
                        y >= chunk_size - lod || (y + lod < chunk_size && !terrain[x][y + lod][z]);
                    let show_bottom = y == 0 || (y >= lod && !terrain[x][y - lod][z]);
                    let show_front =
                        z >= chunk_size - lod || (z + lod < chunk_size && !terrain[x][y][z + lod]);
                    let show_back = z == 0 || (z >= lod && !terrain[x][y][z - lod]);
                    let show_left = x == 0 || (x >= lod && !terrain[x - lod][y][z]);
                    let show_right =
                        x >= chunk_size - lod || (x + lod < chunk_size && !terrain[x + lod][y][z]);

                    let random_number = rng.gen_range(0..=1);
                    let uv_pos: [f32; 2] = [0.0625 * random_number as f32, 0.0];

                    add_cube_with_faces(
                        &mut positions,
                        &mut uvs,
                        &mut normals,
                        &mut tangents,
                        &mut indices,
                        &mut vertex_offset,
                        vec3(x as f32, y as f32, z as f32),
                        lod,
                        show_top,
                        Some(&[
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0],
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0],
                        ]),
                        show_bottom,
                        Some(&[
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0],
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0],
                        ]),
                        show_front,
                        Some(&[
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0],
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0],
                        ]),
                        show_back,
                        Some(&[
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0],
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0],
                        ]),
                        show_left,
                        Some(&[
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0],
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0],
                        ]),
                        show_right,
                        Some(&[
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0625],
                            [uv_pos[0] + 0.0625, uv_pos[1] + 0.0],
                            [uv_pos[0] + 0.0, uv_pos[1] + 0.0],
                        ]),
                    );
                }
            }
        }
    }

    chunk_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    chunk_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    chunk_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    chunk_mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    chunk_mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));

    chunk_mesh
}

fn add_cube_with_faces(
    positions: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    normals: &mut Vec<[f32; 3]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    vertex_offset: &mut usize,
    position: Vec3,
    size: usize,
    show_top: bool,
    top_uvs: Option<&[[f32; 2]]>,
    show_bottom: bool,
    bottom_uvs: Option<&[[f32; 2]]>,
    show_front: bool,
    front_uvs: Option<&[[f32; 2]]>,
    show_back: bool,
    back_uvs: Option<&[[f32; 2]]>,
    show_left: bool,
    left_uvs: Option<&[[f32; 2]]>,
    show_right: bool,
    right_uvs: Option<&[[f32; 2]]>,
) {
    let get_position = |x_offset, y_offset, z_offset| {
        position
            + Vec3::new(
                x_offset as f32 * size as f32,
                y_offset as f32 * size as f32,
                z_offset as f32 * size as f32,
            )
    };

    let compute_tangent =
        |v0: Vec3, v1: Vec3, v2: Vec3, uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2]| {
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;

            let delta_uv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
            let delta_uv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

            let f = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1]);

            let tangent = Vec3::new(
                f * (delta_uv2[1] * edge1.x - delta_uv1[1] * edge2.x),
                f * (delta_uv2[1] * edge1.y - delta_uv1[1] * edge2.y),
                f * (delta_uv2[1] * edge1.z - delta_uv1[1] * edge2.z),
            );

            tangent.normalize()
        };

    // Top face
    if show_top {
        let v0 = get_position(-0.5, 0.5, -0.5);
        let v1 = get_position(0.5, 0.5, -0.5);
        let v2 = get_position(0.5, 0.5, 0.5);
        let v3 = get_position(-0.5, 0.5, 0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(top_uvs.unwrap_or(&[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]));
        normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset,
            offset + 2,
            offset + 1,
            offset,
            offset + 3,
            offset + 2,
        ]);
        *vertex_offset += 4;
    }

    // Bottom face
    if show_bottom {
        let v0 = get_position(-0.5, -0.5, -0.5);
        let v1 = get_position(0.5, -0.5, -0.5);
        let v2 = get_position(0.5, -0.5, 0.5);
        let v3 = get_position(-0.5, -0.5, 0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(bottom_uvs.unwrap_or(&[
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]));
        normals.extend_from_slice(&[[0.0, -1.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset,
            offset + 1,
            offset + 2,
            offset,
            offset + 2,
            offset + 3,
        ]);
        *vertex_offset += 4;
    }

    // Front face
    if show_front {
        let v0 = get_position(-0.5, -0.5, 0.5);
        let v1 = get_position(0.5, -0.5, 0.5);
        let v2 = get_position(0.5, 0.5, 0.5);
        let v3 = get_position(-0.5, 0.5, 0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(front_uvs.unwrap_or(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]));
        normals.extend_from_slice(&[[0.0, 0.0, 1.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset,
            offset + 1,
            offset + 2,
            offset,
            offset + 2,
            offset + 3,
        ]);
        *vertex_offset += 4;
    }

    // Back face
    if show_back {
        let v0 = get_position(0.5, -0.5, -0.5);
        let v1 = get_position(-0.5, -0.5, -0.5);
        let v2 = get_position(-0.5, 0.5, -0.5);
        let v3 = get_position(0.5, 0.5, -0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(back_uvs.unwrap_or(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]));
        normals.extend_from_slice(&[[0.0, 0.0, -1.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset,
            offset + 1,
            offset + 2,
            offset,
            offset + 2,
            offset + 3,
        ]);
        *vertex_offset += 4;
    }

    // Left face
    if show_left {
        let v0 = get_position(-0.5, -0.5, -0.5);
        let v1 = get_position(-0.5, -0.5, 0.5);
        let v2 = get_position(-0.5, 0.5, 0.5);
        let v3 = get_position(-0.5, 0.5, -0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(left_uvs.unwrap_or(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]));
        normals.extend_from_slice(&[[-1.0, 0.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset,
            offset + 1,
            offset + 2,
            offset,
            offset + 2,
            offset + 3,
        ]);
        *vertex_offset += 4;
    }

    // Right face
    if show_right {
        let v0 = get_position(0.5, -0.5, -0.5);
        let v1 = get_position(0.5, -0.5, 0.5);
        let v2 = get_position(0.5, 0.5, 0.5);
        let v3 = get_position(0.5, 0.5, -0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(right_uvs.unwrap_or(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]));
        normals.extend_from_slice(&[[1.0, 0.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset,
            offset + 2,
            offset + 1,
            offset,
            offset + 3,
            offset + 2,
        ]);
        *vertex_offset += 4;
    }
}

#[derive(Component)]
struct FpsRoot;

#[derive(Component)]
struct FpsText;

fn setup_fps_counter(mut commands: Commands) {
    commands
        .spawn((
            FpsRoot,
            NodeBundle {
                background_color: BackgroundColor(Color::BLACK.with_alpha(0.5)),
                z_index: ZIndex::Global(i32::MAX),
                style: Style {
                    position_type: PositionType::Absolute,
                    right: Val::Percent(1.),
                    top: Val::Percent(1.),
                    padding: UiRect::all(Val::Px(4.0)),
                    ..Default::default()
                },
                ..Default::default()
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                FpsText,
                TextBundle::from_sections([
                    TextSection::new(
                        "FPS: ",
                        TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ),
                    TextSection::new(
                        "N/A",
                        TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ),
                ]),
            ));
        });
}

fn fps_text_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed())
        {
            text.sections[1].value = format!("{fps:>4.0}");
            text.sections[1].style.color = match fps {
                fps if fps >= 120.0 => Color::srgb(0.0, 1.0, 0.0),
                fps if fps >= 60.0 => Color::srgb((120.0 - fps as f32) / 60.0, 1.0, 0.0),
                fps if fps >= 30.0 => Color::srgb(1.0, (fps as f32 - 30.0) / 30.0, 0.0),
                _ => Color::srgb(1.0, 0.0, 0.0),
            };
        } else {
            text.sections[1].value = " N/A".into();
            text.sections[1].style.color = Color::WHITE;
        }
    }
}

fn fps_counter_showhide(
    mut q: Query<&mut Visibility, With<FpsRoot>>,
    kbd: Res<ButtonInput<KeyCode>>,
) {
    if kbd.just_pressed(KeyCode::F12) {
        let mut vis = q.single_mut();
        *vis = match *vis {
            Visibility::Hidden => Visibility::Visible,
            _ => Visibility::Hidden,
        };
    }
}
