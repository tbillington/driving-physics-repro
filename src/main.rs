use bevy::prelude::*;
use bevy_xpbd_3d::{prelude::*, PhysicsSchedule, PhysicsStepSet};

fn main() {
    println!("Hello, world!6");
    App::new()
        .insert_resource(SubstepCount(12))
        .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .add_systems(Startup, setup)
        .add_systems(
            PhysicsSchedule,
            (car_update.after(PhysicsStepSet::SpatialQuery),),
        )
        .add_systems(Update, follow_car_2)
        .run();
}

fn follow_car_2(
    car: Query<(&Transform, &LinearVelocity, &Rotation, Entity), (With<Car>, Without<MainCamera>)>,
    mut cam: Query<(&mut Transform, Entity), (With<MainCamera>, Without<Car>)>,
    time: Res<Time>,
    kbd: Res<Input<KeyCode>>,
    mut commands: Commands,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }

    let (car_p, car_l, car_r, car_e) = match car.get_single() {
        Ok(m) => m,
        _ => return,
    };
    let (mut cam, cam_e) = match cam.get_single_mut() {
        Ok(m) => m,
        _ => return,
    };

    let ideal_pos = car_p.translation + Vec3::Y * 2. + car_l.0.normalize_or_zero() * -4.;
    // let offset = ideal_pos.distance(cam.translation);
    // let new_pos = cam.translation
    //     + (ideal_pos - cam.translation)
    //         .clamp_length_max(time.delta_seconds() * car_l.length() * 1.);
    // cam.translation = new_pos;
    cam.translation = ideal_pos;

    let cam_speed = time.delta_seconds() * 5.0;

    let lerp_target = car_p.translation + Vec3::Y * 1.5 + car_l.0.normalize_or_zero() * -1.;
    // println!("{}", lerp_target);

    // cam.translation = cam.translation.lerp(lerp_target, cam_speed);

    let mut cam_with_desired_rot = *cam;
    cam_with_desired_rot.look_at(car_p.translation + Vec3::Y, Vec3::Y);

    cam.rotation = cam.rotation.lerp(cam_with_desired_rot.rotation, cam_speed);
    cam.rotation = cam_with_desired_rot.rotation;

    if kbd.pressed(KeyCode::P) {
        println!("Parenting camera to car");
        commands.entity(cam_e).set_parent_in_place(car_e);
        *done = true;
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Plane
    let plane_size = 64.;
    commands.spawn((
        Name::new("Floor Plane"),
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Plane::from_size(plane_size))),
            material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
            ..default()
        },
        RigidBody::Static,
        Collider::cuboid(plane_size, 0.002, plane_size),
    ));

    commands.spawn((
        Name::new("Sunken Sphere"),
        PbrBundle {
            mesh: meshes.add(
                Mesh::try_from(shape::Icosphere {
                    radius: 10.,
                    subdivisions: 20,
                })
                .unwrap(),
            ),
            transform: Transform::from_xyz(-3., -9., 0.),
            material: materials.add(Color::rgb(0.5, 0.3, 0.3).into()),
            ..default()
        },
        RigidBody::Static,
        Collider::ball(10.),
    ));

    let cube_size = 2.;
    commands.spawn((
        Name::new("Angled Cube"),
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube::new(cube_size))),
            material: materials.add(Color::rgb(0.3, 0.3, 0.5).into()),
            transform: Transform::from_xyz(3., -0.35, 0.)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_6 * 2.)),
            ..default()
        },
        RigidBody::Static,
        Collider::cuboid(cube_size, cube_size, cube_size),
    ));

    //  Car
    let car_settings = CarSettings::PRESET_DRIFT;
    commands.spawn((
        Name::new("Player Car"),
        Car,
        car_settings.clone(),
        CarPhysicsDebug::default(),
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_scale(Vec3::new(1.0, 1.0, 2.0)),
            ..default()
        },
        RigidBody::Dynamic,
        Position(Vec3::Y * 2.0 + Vec3::Z),
        Mass(1.0),
        ExternalForce::ZERO.with_persistence(false),
        Collider::cuboid(1.0, 1.0, 2.0),
    ));

    // Light
    commands.spawn((
        Name::new("Direction Light"),
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                shadows_enabled: true,
                ..default()
            },
            transform: Transform {
                translation: Vec3::new(0.0, 2.0, 0.0),
                rotation: Quat::from_rotation_x(-std::f32::consts::PI / 4.),
                ..default()
            },
            ..default()
        },
        // PointLightBundle {
        //     point_light: PointLight {
        //         intensity: 3500.0,
        //         shadows_enabled: true,
        //         range: 50.,
        //         ..default()
        //     },
        //     transform: Transform::from_xyz(4.0, 8.0, 4.0),
        //     ..default()
        // },
    ));

    // Camera
    commands.spawn((
        MainCamera,
        Camera3dBundle {
            transform: Transform::from_xyz(-4.0, 6.5, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
    ));
}

fn car_update(
    keyboard: Res<Input<KeyCode>>,
    mut cars: Query<
        (
            &mut ExternalForce,
            &Transform,
            Entity,
            &LinearVelocity,
            &AngularVelocity,
            &CarSettings,
            &mut CarPhysicsDebug,
        ),
        With<Car>,
    >,
    spatial_query: SpatialQuery,
) {
    cars.for_each_mut(|(mut ef, p, e, lv, av, car, mut dbg)| {
        let wheels = [
            p.forward() + p.left() * 0.5 + p.down() * 0.5,
            p.forward() + p.right() * 0.5 + p.down() * 0.5,
            p.back() + p.left() * 0.5 + p.down() * 0.5,
            p.back() + p.right() * 0.5 + p.down() * 0.5,
        ];

        dbg.car_speed = p.forward().dot(lv.0);

        wheels.into_iter().enumerate().for_each(|(i, w)| {
            let dbg_wheel = &mut dbg.wheels[i];

            dbg_wheel.offset = w;

            if let Some(hit) = spatial_query.cast_ray(
                p.translation + w,
                p.down() * 1.0,
                0.5,
                true,
                SpatialQueryFilter::default().without_entities([e]),
            ) {
                dbg_wheel.grounded = true;

                // Suspension
                {
                    //point velocity = linear velocity + angular velocity.cross(point - center of mass) (edited)

                    // println!("{:?}", hit);

                    let spring_dir = p.up();

                    let tire_world_vel = lv.0 + av.0.cross(w - car.center_of_mass);
                    dbg_wheel.world_vel = tire_world_vel;

                    let offset = car.spring_rest_dist - hit.time_of_impact;
                    dbg_wheel.offset_from_rest = offset;

                    // println!("offset {offset}");

                    let vel = spring_dir.dot(tire_world_vel);

                    let force = (offset * car.spring_strength) - (vel * car.spring_damper);

                    let force_at_wheel = spring_dir * force;
                    dbg_wheel.suspension_force_at_wheel = force_at_wheel;

                    ef.apply_force_at_point(force_at_wheel, w, car.center_of_mass);
                }

                let steering_influence = if i <= 1 {
                    if keyboard.pressed(KeyCode::A) {
                        p.forward()
                    } else if keyboard.pressed(KeyCode::D) {
                        p.back()
                    } else {
                        Vec3::ZERO
                    }
                } else {
                    Vec3::ZERO
                };

                // Steering
                {
                    let steering_dir = (p.right() + steering_influence).normalize(); // replace this with right after wheel rotation
                    dbg_wheel.steering_dir = steering_dir;

                    // duplicate from suspension?
                    let tire_world_vel = lv.0 + av.0.cross(w - car.center_of_mass);

                    let steering_vel = steering_dir.dot(tire_world_vel);

                    let desired_vel_change = -steering_vel * car.tire_grip_factor;

                    // let desired_accel = desired_vel_change / fixed delta time;

                    let force_at_wheel = steering_dir * (car.tire_mass * desired_vel_change);
                    dbg_wheel.steering_force_at_wheel = force_at_wheel;

                    ef.apply_force_at_point(force_at_wheel, w, car.center_of_mass);
                }

                let steering_influence = if i <= 1 {
                    if keyboard.pressed(KeyCode::A) {
                        p.left()
                    } else if keyboard.pressed(KeyCode::D) {
                        p.right()
                    } else {
                        Vec3::ZERO
                    }
                } else {
                    Vec3::ZERO
                };

                let drive = match car.drive_train {
                    DriveTrain::FrontWheelDrive => i == 0 || i == 1,
                    DriveTrain::RearWheelDrive => i == 2 || i == 3,
                    DriveTrain::AllWheelDrive => true,
                };

                // spread torque across all wheels ?!?

                // Accelleration
                if drive {
                    let accel_dir = (p.forward() + steering_influence).normalize(); // replace this with right after wheel rotation
                    dbg_wheel.accel_dir = accel_dir;

                    if keyboard.pressed(KeyCode::W) || keyboard.pressed(KeyCode::S) {
                        let accel_input = 1.0; // how forward does the user want to go (trigger?)
                        let accel_input = if keyboard.pressed(KeyCode::W) {
                            accel_input
                        } else {
                            -accel_input
                        };

                        // println!("accel_input: {}", accel_input);

                        let car_speed = p.forward().dot(lv.0);

                        let _normalised_speed = (car_speed.abs() / car.top_speed).clamp(0., 1.);

                        // println!("norm speed {}", normalised_speed);

                        let mut torque = car.torque();
                        if accel_input < 0. {
                            torque *= car.reverse_torque_factor;
                        }

                        // let available_torque = powerCurve.Evaluate(normalizedSpeed) * accelInput;
                        let available_torque = torque * accel_input; // do fancy curve stuff here

                        let force_at_wheel = accel_dir * available_torque;
                        dbg_wheel.accel_force_at_wheel = force_at_wheel;

                        ef.apply_force_at_point(force_at_wheel, w, car.center_of_mass);
                    }
                }
            } else {
                dbg_wheel.grounded = false;
            }
        });
    });
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct Car;

#[derive(Debug, Clone, Reflect)]
#[allow(clippy::enum_variant_names)]
enum DriveTrain {
    FrontWheelDrive,
    RearWheelDrive,
    AllWheelDrive,
}

#[derive(Component, Debug, Clone, Reflect)]
struct CarSettings {
    center_of_mass: Vec3,
    drive_train: DriveTrain,

    spring_rest_dist: f32,
    spring_strength: f32,
    spring_damper: f32,

    tire_grip_factor: f32,
    tire_mass: f32,

    top_speed: f32,
    max_torque: f32,
    reverse_torque_factor: f32,
}

impl CarSettings {
    const PRESET_DRIFT: Self = Self {
        center_of_mass: Vec3::new(0., -0.5, 0.),
        drive_train: DriveTrain::AllWheelDrive,

        spring_rest_dist: 0.25,
        spring_strength: 50.5,
        spring_damper: 5.05,

        tire_grip_factor: 0.9,
        tire_mass: 2.0,

        top_speed: 10.,
        max_torque: 15.,
        reverse_torque_factor: 0.25,
    };

    #[inline]
    fn torque(&self) -> f32 {
        use DriveTrain::*;
        match self.drive_train {
            FrontWheelDrive | RearWheelDrive => self.max_torque * 0.5,
            AllWheelDrive => self.max_torque * 0.25,
        }
    }
}

#[derive(Default)]
pub(crate) struct CarPhysicsDebugWheel {
    // suspension
    pub(crate) offset: Vec3,
    pub(crate) grounded: bool,
    pub(crate) world_vel: Vec3,
    pub(crate) offset_from_rest: f32,
    pub(crate) suspension_force_at_wheel: Vec3,

    // steering
    pub(crate) steering_dir: Vec3,
    pub(crate) steering_force_at_wheel: Vec3,

    // accelleration
    pub(crate) accel_dir: Vec3,
    pub(crate) accel_force_at_wheel: Vec3,
}

#[derive(Component, Default)]
pub(crate) struct CarPhysicsDebug {
    pub(crate) wheels: [CarPhysicsDebugWheel; 4],
    pub(crate) car_speed: f32,
}
