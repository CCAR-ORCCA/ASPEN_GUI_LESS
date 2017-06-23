#include "Lidar.hpp"
#include "ShapeModel.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "Filter.hpp"
#include "RK.hpp"
#include "Wrappers.hpp"
#include "Interpolator.hpp"
#include "KDNode.hpp"

#include <chrono>

int main() {

	// Ref frame graph
	FrameGraph frame_graph;
	frame_graph.add_frame("N");
	frame_graph.add_frame("L");
	frame_graph.add_frame("T");
	frame_graph.add_frame("E");

	frame_graph.add_transform("N", "L");
	frame_graph.add_transform("N", "T");
	frame_graph.add_transform("N", "E");

	// Shape model
	ShapeModel true_shape_model("T", &frame_graph);
	ShapeModel estimated_shape_model("E", &frame_graph);

	ShapeModelImporter shape_io_estimated(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/faceted_sphere.obj",
	    200, false);

	ShapeModelImporter shape_io_truth(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_16_scaled.obj", 1);

	shape_io_truth.load_shape_model(&true_shape_model);
	shape_io_estimated.load_shape_model(&estimated_shape_model);

	std::shared_ptr<KDNode> kdtree = std::make_shared<KDNode>(KDNode());
	kdtree = kdtree -> build(*true_shape_model . get_facets(), 0);

	Args args;

	args.set_density(2000);
	args.set_frame_graph(&frame_graph);
	args.set_shape_model( &true_shape_model);

	args.set_e(0.10414291);
	args.set_mu(1.32712440018e+20);
	args.set_sma(149781865609.11);
	args.set_minimum_elevation(20 * arma::datum::pi / 180);



	double t0 = 0;
	double tf = 86400;
	double dt = 0.1; //default timestep. Used as initial guess for RK45


	// 0) Compute ephemeride of HO3

	arma::vec initial_eccentricity = {0};
	RK45 rk_HO3(initial_eccentricity,
	            t0,
	            tf,
	            dt,
	            &args,
	            false,
	            "eccentric_anomaly");


	rk_HO3.run(&E_dot_wrapper,
	           nullptr,
	           nullptr,
	           true);

	arma::mat nu = 2 * arma::atan(std::sqrt( (1 - args.get_e()) / (1 + args.get_e())) * arma::tan(*rk_HO3.get_X() / 2));

	arma::mat s = arma::zeros<arma::mat>( 3, nu.n_cols);
	s.row(0) = arma::cos(nu);
	s.row(1) = arma::sin(nu);

	Interpolator interp_s(rk_HO3.get_T(), &s);

	args.set_interp_s(&interp_s);

	// 1) Propagate small body attitude
	// arma::vec attitude_0 = {0,
	//                         0,
	//                         0,
	//                         0,
	//                         0,
	//                         2 * arma::datum::pi / (12.13 * 3600)
	//                        };

	// bool check_energy_conservation = true;

	// // Specifiying arguments such as density, pointer to frame graph and
	// // attracting shape model

	// args.set_is_attitude_bool(true);

	// RK45 rk_attitude(attitude_0,
	//                  t0,
	//                  tf,
	//                  dt,
	//                  &args,
	//                  check_energy_conservation,
	//                  "attitude");

	// rk_attitude.run(&attitude_dxdt_wrapper,
	//                 &energy_attitude,
	//                 &event_function_mrp);

	// // 2) Propagate spacecraft attitude about small body
	// // using computed small body attitude
	// Interpolator interpolator(rk_attitude . get_T(), rk_attitude . get_X());
	// args.set_interpolator(&interpolator);
	// args.set_is_attitude_bool(false);

	// // Initial condition of the orbiting spacecraft
	// arma::vec initial_pos = {800, 0, 0};
	// arma::vec initial_vel_inertial = {0, 0.0, 0.06};

	// arma::vec body_vel = initial_vel_inertial - arma::cross(attitude_0.rows(3, 5),
	//                      initial_pos);

	// arma::vec orbit_0(6);

	// orbit_0.rows(0, 2) = initial_pos;
	// orbit_0.rows(3, 5) = body_vel;

	// RK45 rk_orbit( orbit_0,
	//                t0,
	//                tf,
	//                dt,
	//                &args,
	//                check_energy_conservation,
	//                "orbit_body_frame"
	//              );

	// rk_orbit.run(&pgm_dxdt_wrapper_body_frame,
	//              &energy_orbit_body_frame,
	//              &event_function_collision_body_frame,
	//              true);

	// // The attitude of the asteroid is also interpolated
	// arma::mat interpolated_attitude = arma::mat(6, rk_orbit.get_T() -> n_rows);

	// for (unsigned int i = 0; i < interpolated_attitude.n_cols; ++i) {
	// 	interpolated_attitude.col(i) = interpolator.interpolate( rk_orbit.get_T() -> at(i), true);
	// }
	// interpolated_attitude.save("interpolated_attitude.txt", arma::raw_ascii);
	// throw;
	// Lidar
	Lidar lidar(&frame_graph, "L", 10, 10 , 32, 32, 1e-2, 1. / 3600, &args,kdtree.get());

	// Instrument orbit (rate and inclination)
	// double orbit_rate =  2 * arma::datum::pi * 1e-2 ;
	// double inclination = 80 * arma::datum::pi / 180;

	// Target angular rate
	// double body_spin_rate =  2e-1 ;

	// // Minimum impact angle for a measurement to be used (deg)
	// double min_normal_observation_angle = 10 * arma::datum::pi / 180.;

	// // Minimum angular difference between two neighboring surface
	// // normals to be considered different
	// double min_facet_normal_angle_difference = 45 * arma::datum::pi / 180.;

	// // Minimum number of rays required for a facet to be considered
	// // in the estimation process
	// unsigned int minimum_ray_per_facet = 5;

	// // Maximum number of times a facet and its children can be split
	// unsigned int max_split_count = 5;

	// // Ridge estimation
	// double ridge_coef = 1e1;

	// // Remove outliers
	// bool reject_outliers = true;

	// // Activate surface splitting
	// bool split_status = true;

	// // Use cholesky decomposition (WARNING: SHOULD TURN OFF RIDGE IF TRUE)
	// bool use_cholesky = false;

	// // Recycle degenerate facets
	// bool recycle_shrunk_facets = true;

	// // Minimum facet angle indicating degeneracy
	// double min_facet_angle = 10 * arma::datum::pi / 180;

	// // Minimum edge angle indicating degeneracy
	// // 0 deg: no edge recycling at all
	// // 90 deg : facets cannot have their normals in opposite directions
	// double min_edge_angle = 20 * arma::datum::pi / 180;

	// // Filter arguments
	// FilterArguments args = FilterArguments( t0,
	//                             tf,
	//                             min_normal_observation_angle,
	//                             orbit_rate,
	//                             inclination,
	//                             body_spin_rate,
	//                             min_facet_normal_angle_difference,
	//                             ridge_coef,
	//                             min_facet_angle,
	//                             min_edge_angle,
	//                             minimum_ray_per_facet,
	//                             max_split_count,
	//                             reject_outliers,
	//                             split_status,
	//                             use_cholesky,
	//                             recycle_shrunk_facets);



	// Filter
	Filter filter(&frame_graph,
	              &lidar,
	              &true_shape_model);

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	// filter.get_surface_point_cloud_from_trajectory(rk_orbit.get_X(),
	//         rk_orbit.get_T(),
	//         rk_attitude.get_X(),
	//         rk_attitude.get_T(),
	//         "itokawa_pc.obj");

	filter.get_surface_point_cloud_from_trajectory(
	    "../build/X_RK45_orbit_body_frame.txt",
	    "../build/T_RK45_orbit_body_frame.txt",
	    "../build/X_RK45_attitude.txt",
	    "../build/T_RK45_attitude.txt",
	    "itokawa_pc.obj");

	true_shape_model. save_lat_long_map_to_file("lat_long_impacts.txt");

	// end = std::chrono::system_clock::now();
	// std::chrono::duration<double> elapsed_seconds = end - start;
	// std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	// // filter.run(5, true, true);



	return 0;
}












