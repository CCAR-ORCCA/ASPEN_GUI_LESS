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
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/KW4Alpha.obj",
	    1);

	shape_io_truth.load_shape_model(&true_shape_model);
	shape_io_estimated.load_shape_model(&estimated_shape_model);

	std::shared_ptr<KDNode> kdtree = std::make_shared<KDNode>(KDNode());
	kdtree = kdtree -> build(*true_shape_model . get_facets(), 0);

	// Lidar
	arma::vec lidar_pos = { 2, 0., 0.};
	frame_graph.set_transform_origin("N", "L", lidar_pos);
	Lidar lidar(&frame_graph, "L", 10, 10, 128, 128, 1e-2, 1, kdtree.get());

	// Instrument orbit (rate and inclination)
	double orbit_rate =  2 * arma::datum::pi * 1e-2 ;
	double inclination = 80 * arma::datum::pi / 180;

	// Target angular rate
	double body_spin_rate =  2e-1 ;

	// Minimum impact angle for a measurement to be used (deg)
	double min_normal_observation_angle = 10 * arma::datum::pi / 180.;

	// Minimum angular difference between two neighboring surface
	// normals to be considered different
	double min_facet_normal_angle_difference = 45 * arma::datum::pi / 180.;

	// Minimum number of rays required for a facet to be considered
	// in the estimation process
	unsigned int minimum_ray_per_facet = 5;

	// Maximum number of times a facet and its children can be split
	unsigned int max_split_count = 5;

	// Ridge estimation
	double ridge_coef = 1e1;

	// Remove outliers
	bool reject_outliers = true;

	// Activate surface splitting
	bool split_status = true;

	// Use cholesky decomposition (WARNING: SHOULD TURN OFF RIDGE IF TRUE)
	bool use_cholesky = false;

	// Recycle degenerate facets
	bool recycle_shrunk_facets = true;

	// Minimum facet angle indicating degeneracy
	double min_facet_angle = 10 * arma::datum::pi / 180;

	// Minimum edge angle indicating degeneracy
	// 0 deg: no edge recycling at all
	// 90 deg : facets cannot have their normals in opposite directions
	double min_edge_angle = 20 * arma::datum::pi / 180;

	// Time spans
	double t0 = 0;
	double tf = 100;

	// Filter arguments
	Arguments args = Arguments( t0,
	                            tf,
	                            min_normal_observation_angle,
	                            orbit_rate,
	                            inclination,
	                            body_spin_rate,
	                            min_facet_normal_angle_difference,
	                            ridge_coef,
	                            min_facet_angle,
	                            min_edge_angle,
	                            minimum_ray_per_facet,
	                            max_split_count,
	                            reject_outliers,
	                            split_status,
	                            use_cholesky,
	                            recycle_shrunk_facets);


	// // Filter
	Filter filter(&frame_graph,
	              &lidar,
	              &true_shape_model,
	              &estimated_shape_model,
	              &args);

	// filter.run(5, true, true);




	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();



	filter.get_surface_point_cloud("KW4_kd.obj");



	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";


	// 1) Propagate small body attitude
	// arma::vec attitude_0 = {0,
	//                         0,
	//                         0,
	//                         0.000,
	//                         0.000,
	//                         2 * arma::datum::pi / (12.13 * 3600)
	//                        };
	// double t0 = 0;
	// double tf = 2 * 86400;
	// double dt = 1;

	// bool check_energy_conservation = true;


	// Args args(2000,
	//           &frame_graph,
	//           &true_shape_model);


	// RK45 rk_attitude(attitude_0,
	//                  t0,
	//                  tf,
	//                  dt,
	//                  &args,
	//                  check_energy_conservation,
	//                  "attitude");

	// rk_attitude.run(&attitude_dxdt_wrapper, &energy_attitude, &event_function_mrp);


	// // 2) Propagate spacecraft attitude about small body
	// // using computed small body attitude
	// Interpolator interpolator(rk_attitude . get_T(), rk_attitude . get_X());
	// args.set_interpolator(&interpolator);

	// arma::vec orbit_0 = {1000, 0, 0, -0.005, 0.0, 0.035};

	// // dt = 1;
	// RK45 rk_orbit( orbit_0,
	//                t0,
	//                tf,
	//                dt,
	//                &args,
	//                check_energy_conservation,
	//                "orbit"
	//              );

	// rk_orbit.run(&pgm_dxdt_wrapper, &energy_orbit, &event_function_collision, true);

	// // The attitude of the asteroid is also interpolated
	// arma::mat interpolated_attitude = arma::mat(6, rk_orbit.get_T() -> n_rows);

	// for (unsigned int i = 0; i < interpolated_attitude.n_cols; ++i) {
	// 	interpolated_attitude.col(i) = interpolator.interpolate( rk_orbit.get_T() -> at(i));
	// }
	// interpolated_attitude.save("interpolated_attitude.txt", arma::raw_ascii);


	return 0;
}












