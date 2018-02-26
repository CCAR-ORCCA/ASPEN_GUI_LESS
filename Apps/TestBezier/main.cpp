#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"


#include <chrono>


int main(){

	FrameGraph frame_graph;
	
	ShapeModelTri test_tri("", &frame_graph);
	ShapeModelBezier test("", &frame_graph);


	// ShapeModelImporter shape_io("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/shape_model/fit_source_400.b", 1, true);
	// ShapeModelImporter shape_io_tri("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/shape_model/fit_source_400.obj", 1, true);
		

	ShapeModelImporter shape_io("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/patch.b", 1, true);
	// ShapeModelImporter shape_io_tri("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/cube.obj", 1, true);
	

	shape_io.load_bezier_shape_model(&test);
	// shape_io_tri.load_obj_shape_model(&test_tri);


	std::cout << test.get_center_of_mass() << std::endl;
	// std::cout << test_tri.get_center_of_mass() << std::endl;


	return 0;
}