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


	ShapeModelImporter::load_obj_shape_model("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/true_shape_L0.obj"
		, 1, true,truth);


	ShapeModelImporter::load_bezier_shape_model("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/fit_shape.b"
		, 1, true,bezier);



	std::cout << truth.get_center_of_mass().t();
	std::cout << bezier.get_center_of_mass().t();

	return 0;
}