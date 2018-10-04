#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "RigidBodyKinematics.hpp"



#include <chrono>


int main(){

	
	FrameGraph frame_graph;
	
	ShapeModelBezier bezier("", &frame_graph);
	ShapeModelImporter io("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/cube.b", 1, true);
	io.load_bezier_shape_model(&bezier);

	std::cout << bezier.get_volume() << std::endl;
	std::cout << bezier.get_inertia() << std::endl;
	std::cout << bezier.get_center_of_mass() << std::endl;

	return 0;
}