use bevy::{diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin}, math::vec3, pbr::*, prelude::*, render::{mesh::*, render_asset::*}, window::{PresentMode, WindowTheme}};
use bevy_flycam::{NoCameraPlayerPlugin, FlyCam};
use noise::{NoiseFn, Perlin};
use std::f32::consts::PI;

fn main() {
    App::new()
    .add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Voxel3d".into(),
                name: Some("cringekitsune.voxel3d.game".into()),
                resolution: (1080., 720.).into(),
                present_mode: PresentMode::AutoNoVsync,
                window_theme: Some(WindowTheme::Dark),
                ..Default::default()
            }),
            ..Default::default()
        }).set(ImagePlugin::default_nearest()),
        NoCameraPlayerPlugin,
        FrameTimeDiagnosticsPlugin
    ))
    .add_systems(Startup, (setup, spawn_terrain, setup_fps_counter))
    .add_systems(Update, (fps_text_update_system, fps_counter_showhide))
    .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        },
        FlyCam,
    ));

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            PI / 2.,
            -PI / 4.,
        )),
        cascade_shadow_config: CascadeShadowConfigBuilder {
            first_cascade_far_bound: 7.0,
            maximum_distance: 64.0,
            ..default()
        }
        .into(),
        ..default()
    });

}

fn spawn_terrain(
    mut commands: Commands,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let custom_texture_handle: Handle<Image> = asset_server.load("textures/texture.png");

    let cube_material = material_assets.add(StandardMaterial {
        base_color_texture: Some(custom_texture_handle),
        ..Default::default()
    });


    let chunk_size = 256;
    let perlin = Perlin::new(192);
    let scale = 0.055;

    let mut chunk_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD);
    
    let mut terrain = vec![vec![vec![false; chunk_size]; chunk_size]; chunk_size];

    for x in 0..chunk_size {
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                let noise_value = perlin.get([x as f64 * scale, y as f64 * scale, z as f64 * scale]);
                if noise_value > 0.3 {
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

    for x in 0..chunk_size {
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                if terrain[x][y][z] {
                    let show_top = y == chunk_size - 1 || !terrain[x][y + 1][z]; // No voxel above
                    let show_bottom = y == 0 || !terrain[x][y - 1][z]; // No voxel below
                    let show_front = z == chunk_size - 1 || !terrain[x][y][z + 1]; // No voxel in front
                    let show_back = z == 0 || !terrain[x][y][z - 1]; // No voxel behind
                    let show_left = x == 0 || !terrain[x - 1][y][z]; // No voxel to the left
                    let show_right = x == chunk_size - 1 || !terrain[x + 1][y][z]; // No voxel to the right

                    // Only add visible faces
                    add_cube_with_faces(
                        &mut positions,
                        &mut uvs,
                        &mut normals,
                        &mut tangents,
                        &mut indices,
                        &mut vertex_offset,
                        vec3(x as f32, y as f32, z as f32),
                        show_top,
                        show_bottom,
                        show_front,
                        show_back,
                        show_left,
                        show_right,
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

    // Spawn chunk mesh
    commands.spawn(PbrBundle {
        mesh: mesh_assets.add(chunk_mesh),
        material: cube_material.clone(),
        transform: Transform::from_translation(Vec3::ZERO),
        ..Default::default()
    });
}

fn add_cube_with_faces(
    positions: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    normals: &mut Vec<[f32; 3]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    vertex_offset: &mut usize,
    position: Vec3,
    show_top: bool,
    show_bottom: bool,
    show_front: bool,
    show_back: bool,
    show_left: bool,
    show_right: bool,
) {
    let get_position = |x_offset, y_offset, z_offset| position + Vec3::new(x_offset, y_offset, z_offset);

    let compute_tangent = |v0: Vec3, v1: Vec3, v2: Vec3, uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2]| {
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

    if show_top {
        let v0 = get_position(-0.5, 0.5, -0.5);
        let v1 = get_position(0.5, 0.5, -0.5);
        let v2 = get_position(0.5, 0.5, 0.5);
        let v3 = get_position(-0.5, 0.5, 0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(&[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset, offset + 2, offset + 1,
            offset, offset + 3, offset + 2,
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
        uvs.extend_from_slice(&[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        normals.extend_from_slice(&[[0.0, -1.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset, offset + 1, offset + 2,
            offset, offset + 2, offset + 3,
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
        uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        normals.extend_from_slice(&[[0.0, 0.0, 1.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset, offset + 1, offset + 2,
            offset, offset + 2, offset + 3,
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
        uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        normals.extend_from_slice(&[[0.0, 0.0, -1.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset, offset + 1, offset + 2,
            offset, offset + 2, offset + 3,
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
        uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        normals.extend_from_slice(&[[-1.0, 0.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset, offset + 1, offset + 2,
            offset, offset + 2, offset + 3,
        ]);        
        *vertex_offset += 4;
    }

    // Right face
    if show_right {
        let v0 = get_position(0.5, -0.5, 0.5);
        let v1 = get_position(0.5, -0.5, -0.5);
        let v2 = get_position(0.5, 0.5, -0.5);
        let v3 = get_position(0.5, 0.5, 0.5);

        positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
        uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        normals.extend_from_slice(&[[1.0, 0.0, 0.0]; 4]);

        let tangent = compute_tangent(v0, v1, v2, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        tangents.extend_from_slice(&[[tangent.x, tangent.y, tangent.z, 0.0]; 4]);

        let offset = *vertex_offset as u32;
        indices.extend_from_slice(&[
            offset, offset + 1, offset + 2,
            offset, offset + 2, offset + 3,
        ]);
        *vertex_offset += 4;
    }
}

#[derive(Component)]
struct FpsRoot;

#[derive(Component)]
struct FpsText;

fn setup_fps_counter(
    mut commands: Commands,
) {
    commands.spawn((
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
                TextSection::new("FPS: ", TextStyle { font_size: 16.0, color: Color::WHITE, ..default() }),
                TextSection::new("N/A", TextStyle { font_size: 16.0, color: Color::WHITE, ..default() }),
            ]),
        ));
    });
}

fn fps_text_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(|d| d.smoothed()) {
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