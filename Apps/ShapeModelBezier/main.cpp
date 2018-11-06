#include "Bezier.hpp"
#include "Facet.hpp"
#include "ControlPoint.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeFitterBezier.hpp"




#include <chrono>
int main(){
	FrameGraph frame_graph;

	ShapeModelTri<ControlPoint> truth("", nullptr);
	ShapeModelBezier<ControlPoint> bezier("", nullptr);


	ShapeModelImporter::load_obj_shape_model("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/fit_shape.obj"
		, 1, true,truth);


	ShapeModelImporter::load_bezier_shape_model("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/fit_shape.b"
		, 1, true,bezier);




	return 0;
}