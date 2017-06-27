#include "Lidar.hpp"
#include "ShapeModel.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "Filter.hpp"
#include "RK.hpp"
#include "Wrappers.hpp"
#include "Interpolator.hpp"
#include "KDNode.hpp"
#include "Constants.hpp"

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
	// ShapeModel estimated_shape_model("E", &frame_graph);

	// ShapeModelImporter shape_io_estimated(
	//     "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/faceted_sphere.obj",
	//     200, false);

	ShapeModelImporter shape_io_truth(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/HO3_scaled.obj", 1);

	shape_io_truth.load_shape_model(&true_shape_model);
	// shape_io_estimated.load_shape_model(&estimated_shape_model);

	std::shared_ptr<KDNode> kdtree = std::make_shared<KDNode>(KDNode());
	kdtree = kdtree -> build(*true_shape_model . get_facets(), 0);

	Args args;

	args.set_frame_graph(&frame_graph);
	args.set_shape_model( &true_shape_model);

	args.set_e(ECCENTRICITY);
	args.set_mu(MU);
	args.set_sma(SMA);
	args.set_i(INCLINATION);
	args.set_Omega(HO3_RIGHT_ASCENSION);
	args.set_omega(HO3_LONGITUDE_PERIGEE);
	args.set_minimum_elevation(MINIMUM_ELEVATION * arma::datum::pi / 180);


	// 0) Load ephemeride of HO3
	arma::mat nu;
	arma::vec times;
	nu.load("/Users/bbercovici/Desktop/Nicolas_code/nu.txt");
	times.load("/Users/bbercovici/Desktop/Nicolas_code/time.txt");

	arma::inplace_trans(nu);

	arma::mat s = arma::zeros<arma::mat>( 3, nu.n_cols);
	s.row(0) = arma::cos(nu + args.get_omega());
	s.row(1) = arma::sin(nu + args.get_omega());
	s = (M3(- args.get_Omega()) * M1(- args.get_i())) * s;

	Interpolator interp_s(&times, &s);
	args.set_interp_s(&interp_s);

	Lidar lidar(&frame_graph, "L", ROW_FOV, COL_FOV , ROW_RESOLUTION,
	            COL_RESOLUTION, FOCAL_LENGTH, INSTRUMENT_FREQUENCY, &args, kdtree.get());


	// Filter
	Filter filter(&frame_graph,
	              &lidar,
	              &true_shape_model);

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();


	filter.get_surface_point_cloud_from_trajectory(
	    "/Users/bbercovici/Desktop/Nicolas_code/Trajectory_BodyFixed.txt",
	    "HO3_pc.obj");

	true_shape_model. save_lat_long_map_to_file("lat_long_impacts.txt");

	// end = std::chrono::system_clock::now();
	// std::chrono::duration<double> elapsed_seconds = end - start;
	// std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	// // filter.run(5, true, true);



	return 0;
}












