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

	arma::vec lidar_pos = { -3., 0., 0.};
//

	// arma::vec lidar_pos = { -0.7648  , -0.6442  ,      0};
	// arma::vec lidar_mrp = { 0      ,  0  , 0.1768};


	frame_graph.set_transform_origin("N", "L", lidar_pos);
	// frame_graph.set_transform_mrp("N", "L", lidar_mrp);


	// Shape model
	ShapeModel true_shape_model("T", &frame_graph);
	ShapeModel estimated_shape_model("E", &frame_graph);

	ShapeModelImporter shape_io_truth("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/tetra.obj",
	                                  1);
	ShapeModelImporter shape_io_estimated("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/tetra.obj",
	                                      0.5);

	shape_io_truth.load_shape_model(&true_shape_model);
	shape_io_estimated.load_shape_model(&estimated_shape_model);

	// Lidar
	Lidar lidar(&frame_graph, "L", 55, 55, 32, 32, 1e-2, 1);

	// Angular rate of the instrument about the target (rad/s)

	double omega = 2 * arma::datum::pi / (  20);

	// Filter
	Filter filter(&frame_graph, &lidar, &true_shape_model, &estimated_shape_model, 0,  20, omega, arma::datum::pi * 45. / 180.);
	filter.run(10, false, true);


	return 0;
}












