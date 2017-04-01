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

	arma::vec lidar_pos = { -5., 0., 0.};
	frame_graph.set_transform_origin("N", "L", lidar_pos);

	// Shape model
	ShapeModel true_shape_model("T", &frame_graph);
	ShapeModel estimated_shape_model("E", &frame_graph);

	ShapeModelImporter shape_io_truth(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/cube.obj",
	    1);
	ShapeModelImporter shape_io_estimated(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/cube.obj",
	    0.3);

	shape_io_truth.load_shape_model(&true_shape_model);
	shape_io_estimated.load_shape_model(&estimated_shape_model);

	// Lidar
	Lidar lidar(&frame_graph, "L", 15, 15, 32, 32, 1e-2, 1);

	// Angular rate of the instrument about the target (rad/s)
	double omega = 2 * arma::datum::pi / (  20);

	// Minimum impact angle for a measurement to be used (deg)
	double min_normal_observation_angle = 60;
	double min_facet_normal_angle_difference = 10;

	// Minimum number of rays required for a facet to be considered
	// in the estimation process
	unsigned int minimum_ray_per_facet = 10;

	// Ridge estimation
	double ridge_coef = 0;

	// Filter
	Filter filter(&frame_graph,
	              &lidar,
	              &true_shape_model,
	              &estimated_shape_model,
	              0,
	              40,
	              omega,
	              arma::datum::pi * min_normal_observation_angle / 180.,
	              arma::datum::pi * min_facet_normal_angle_difference / 180.,
	              minimum_ray_per_facet,
	              ridge_coef);

	filter.run(3, true, true);

	return 0;
}












