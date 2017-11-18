#include "Bezier.hpp"
#include "Facet.hpp"
#include "PC.hpp"
#include "ControlPoint.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelImporter.hpp"



#include <chrono>
int main(){


	FrameGraph frame_graph;
	
	// Shape model formed with triangles
	ShapeModelTri shape_tri("", &frame_graph);

	
	ShapeModelImporter shape_io(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_64.obj", 1000, true);

	shape_io.load_shape_model(&shape_tri);
	shape_tri.save("tri.obj");

	ShapeModelBezier shape_bezier(&shape_tri,
		"",
		&frame_graph);

	unsigned int D = 2;
	for (unsigned int i = 1; i < D + 1; ++i){
		std::cout << i << std::endl;
		shape_bezier.elevate_n();
	}
	shape_bezier.save("bezier_" + std::to_string(D)+ ".obj");







	return 0;
}