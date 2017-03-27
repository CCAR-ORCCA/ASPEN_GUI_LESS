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

	arma::vec lidar_pos = { -1500., 0., 0.};
	frame_graph.set_transform_origin("N", "L", lidar_pos);

	// Shape model
	ShapeModel true_shape_model("T", &frame_graph);
	ShapeModel estimated_shape_model("E", &frame_graph);

	ShapeModelImporter shape_io_truth("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_16.obj",
	                            1000);
	ShapeModelImporter shape_io_estimated("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_8.obj",
	                            1000);

	shape_io_truth.load_shape_model(&true_shape_model);
	shape_io_estimated.load_shape_model(&estimated_shape_model);


	// Lidar
	Lidar lidar(&frame_graph, "L", 15, 15, 32, 32, 1e-2, 1);

	// Filter
	Filter filter(&frame_graph, &lidar, &true_shape_model, &estimated_shape_model, 0, 100, 1e-3);
	filter.run();

	/**
	Need a Filter class to host:
		- the instrument
		- the true shape model
		- the a-priori shape model
		- Filtering tools:
			# the partial derivatives evaluation
			# shape refinement

	Should also augment the lidar and/or the shape model with a "Dynamics"
	member that enables propagation in time.
	All I really need is just to actualize the reference frame corresponding to each

	This can be done by querying the reference frame of interest in FrameGraph
	and setting its mrp/origin with respect to its parent
	*/

	return 0;
}












