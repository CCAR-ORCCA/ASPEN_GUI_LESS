#include "ControlPoint.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelImporter.hpp"

#include <chrono>


int main(){

	FrameGraph frame_graph;
	
	// Shape model formed with triangles
	ShapeModelTri shape_tri("", &frame_graph);

	ShapeModelImporter shape_io(
		"bennu.obj", 1, true);
	shape_io.load_obj_shape_model(&shape_tri);
	
	ShapeModelBezier bennu(&shape_tri,
		"",
		&frame_graph);
	bennu.elevate_degree();
	bennu.save("bennu.b");
	bennu.save_to_obj("bennu_saved.obj");

	ShapeModelImporter shape_io_2(
		"bennu.b", 1, true);
	ShapeModelBezier bennu_2("",
		&frame_graph);
	shape_io_2.load_bezier_shape_model(&bennu_2);

	bennu_2.save_to_obj("bennu_2.obj");




	



	return 0;
}