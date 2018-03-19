#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "RigidBodyKinematics.hpp"



#include <chrono>


int main(){

	FrameGraph frame_graph;
	
	ShapeModelTri itokawa_true("", &frame_graph);
	ShapeModelBezier itokawa_bezier_fit("", &frame_graph);


	ShapeModelImporter shape_io_bezier("../../../resources/shape_models/cube_2.b", 1, true);
	ShapeModelImporter shape_io_true("../../../resources/shape_models/cube.obj", 1, true);
	

	// ShapeModelImporter shape_io_bezier("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/shape_model/test/fit_shape_aligned.b", 1, true);
	// ShapeModelImporter shape_io_true("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/shape_model/test/fit_shape_aligned.obj", 1, true);
	
	shape_io_bezier.load_bezier_shape_model(&itokawa_bezier_fit);
	shape_io_true.load_obj_shape_model(&itokawa_true);

	std::cout << "\nVolume: \n";

	std::cout << itokawa_bezier_fit.get_volume() << std::endl;
	std::cout << itokawa_true.get_volume() << std::endl;

	std::cout << "\nCenter of mass: \n";
	
	std::cout << itokawa_bezier_fit.get_center_of_mass() << std::endl;
	std::cout << itokawa_true.get_center_of_mass() << std::endl;

	std::cout << "\nInertia: \n";
	
	std::cout << itokawa_bezier_fit.get_inertia() << std::endl;
	std::cout << itokawa_true.get_inertia() << std::endl;



	return 0;
}