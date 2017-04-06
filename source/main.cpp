#include "Lidar.hpp"
#include "ShapeModel.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "Filter.hpp"

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

	// ShapeModelImporter shape_io_truth(
	//     "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_8.obj",
	//     1000);

	// ShapeModelImporter shape_io_estimated(
	//     "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/faceted_sphere.obj",
	//     100);


	ShapeModelImporter shape_io_truth(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/KW4Alpha.obj",
	    1);

	ShapeModelImporter shape_io_estimated(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/faceted_sphere.obj",
	    400);



	shape_io_truth.load_shape_model(&true_shape_model);
	shape_io_estimated.load_shape_model(&estimated_shape_model);

	// Lidar
	// arma::vec lidar_pos = { 1000, 0., 0.};

	arma::vec lidar_pos = { 2500, 0., 0.};
	frame_graph.set_transform_origin("N", "L", lidar_pos);
	Lidar lidar(&frame_graph, "L", 25, 25, 32, 32, 1e-2, 1);

	// Instrument orbit (rate and inclination)
	double orbit_rate =  arma::datum::pi * 1e-2 ;
	double inclination = 80 * arma::datum::pi / 180;

	// Target angular rate
	double body_spin_rate =  1e-1 ;

	// Minimum impact angle for a measurement to be used (deg)
	double min_normal_observation_angle = 10 * arma::datum::pi / 180.;

	// Minimum angular difference between two neighboring surface
	// normals to be considered different
	double min_facet_normal_angle_difference = 30 * arma::datum::pi / 180.;

	// Minimum number of rays required for a facet to be considered
	// in the estimation process
	unsigned int minimum_ray_per_facet = 5;

	// Ridge estimation
	double ridge_coef = 1e1;

	// Remove outliers
	bool reject_outliers = true;

	// Activate surface splitting
	bool split_status = true;

	// Use cholesky decomposition (WARNING: SHOULD TURN OFF RIDGE IF TRUE)
	bool use_cholesky = false;

	// Recycle degenerate facets
	bool recycle_facets = true;

	// Time spans
	double t0 = 0;
	double tf = 0;

	// Filter arguments
	Arguments args = Arguments( t0,
	                            tf,
	                            min_normal_observation_angle,
	                            orbit_rate,
	                            inclination,
	                            body_spin_rate,
	                            min_facet_normal_angle_difference,
	                            minimum_ray_per_facet,
	                            ridge_coef,
	                            reject_outliers,
	                            split_status,
	                            use_cholesky,
	                            recycle_facets);

	// Filter
	Filter filter(&frame_graph,
	              &lidar,
	              &true_shape_model,
	              &estimated_shape_model,
	              &args);

	filter.run(10, true, true);





	return 0;
}












