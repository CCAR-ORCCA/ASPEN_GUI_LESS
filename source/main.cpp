#include "Lidar.hpp"
#include "ShapeModel.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "Filter.hpp"

int main() {
	// Ref frame graph
	FrameGraph frame_graph;
	frame_graph.add_frame("N");
	frame_graph.add_frame("L");
	frame_graph.add_frame("T");
	frame_graph.add_transform("N", "L");
	frame_graph.add_transform("N", "T");

	arma::vec lidar_pos = { -3000., 0., 0.};
	frame_graph.set_transform_origin("N", "L", lidar_pos);
	arma::vec target_mrp = {0.1, 0, 0};
	frame_graph.set_transform_mrp("N", "T", target_mrp);

	// Shape model
	ShapeModel shape_model("T", &frame_graph);
	ShapeModelImporter shape_io("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/KW4Alpha.obj");
	shape_io.load_shape_model(&shape_model);

	// Lidar
	Lidar lidar(&frame_graph, "L", 30, 30, 32, 32, 1e-2);
	lidar.send_flash(&shape_model);
	lidar.save_focal_plane_range("ranges.txt");

	// Filter
	Filter filter(&lidar, &shape_model, &shape_model,0,10,1e-3);

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












