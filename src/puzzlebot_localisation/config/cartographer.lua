-- Cartographer 2D configuration for Puzzlebot (RPLidar A1/A2, wheel odometry)
--
-- TF tree expected:
--   odom → base_footprint   (published by localisation.py)
--
-- Cartographer publishes:
--   map → odom              (corrected, with loop closure)
--   /map                    (OccupancyGrid via cartographer_occupancy_grid_node)

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder                               = MAP_BUILDER,
  trajectory_builder                        = TRAJECTORY_BUILDER,

  -- Frames
  map_frame                                 = "map",
  tracking_frame                            = "base_footprint",
  -- published_frame is the child of the transform Cartographer outputs.
  -- With provide_odom_frame=false the output is map→odom.
  published_frame                           = "odom",
  odom_frame                                = "odom",
  provide_odom_frame                        = false,   -- localisation.py provides odom→base_footprint
  publish_frame_projected_to_2d             = true,

  -- Sensors
  use_pose_extrapolator                     = true,
  use_odometry                              = true,    -- /odom from wheel encoders
  use_nav_sat                               = false,
  use_landmarks                             = false,
  num_laser_scans                           = 1,
  num_multi_echo_laser_scans                = 0,
  num_subdivisions_per_laser_scan           = 1,
  num_point_clouds                          = 0,

  -- Timing / latency
  lookup_transform_timeout_sec              = 0.3,
  submap_publish_period_sec                 = 0.3,
  pose_publish_period_sec                   = 5e-3,
  trajectory_publish_period_sec             = 30e-3,

  -- Sampling ratios (keep all data)
  rangefinder_sampling_ratio                = 1.,
  odometry_sampling_ratio                   = 1.,
  fixed_frame_pose_sampling_ratio           = 1.,
  imu_sampling_ratio                        = 1.,
  landmarks_sampling_ratio                  = 1.,
}

-- ── 2D map builder ───────────────────────────────────────────────────────────
MAP_BUILDER.use_trajectory_builder_2d = true

-- ── Local trajectory builder (scan matching) ─────────────────────────────────
TRAJECTORY_BUILDER_2D.min_range                     = 0.12   -- RPLidar blind zone
TRAJECTORY_BUILDER_2D.max_range                     = 4.5    -- clip to indoor range
TRAJECTORY_BUILDER_2D.missing_data_ray_length       = 1.0
TRAJECTORY_BUILDER_2D.use_imu_data                  = false

-- Submaps: how many laser scans form one submap.
-- Smaller → more submaps → finer loop closure; larger → smoother local map.
TRAJECTORY_BUILDER_2D.submaps.num_range_data        = 35
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.01  -- m/cell (matches icp_map)

-- Real-time correlative scan matcher (CSM) — runs before Ceres for a quick
-- global search. Helps on low-odometry-accuracy robots.
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching               = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window  = 0.10  -- m
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 1e-1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight    = 1e-1

-- Ceres scan matcher — fine alignment after CSM
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 2.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight    = 40.

-- Motion filter: skip scans when the robot hasn't moved much (reduces CPU).
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds        = 0.5
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters     = 0.1
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians       = math.rad(0.1)

-- ── Pose graph (global SLAM / loop closure) ──────────────────────────────────
POSE_GRAPH.optimize_every_n_nodes = 35   -- run graph optimisation frequently

-- Constraint builder: how strict loop closure matching must be.
POSE_GRAPH.constraint_builder.min_score                       = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score   = 0.70
POSE_GRAPH.constraint_builder.sampling_ratio                  = 0.3

-- Search window for loop closure candidates
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window  = 3.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(45.)

return options
