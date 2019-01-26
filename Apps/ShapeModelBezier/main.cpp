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

	ShapeModelImporter::load_obj_shape_model("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_8.obj"
		, 1, true,truth);


	ShapeModelBezier<ControlPoint> bezier(truth,"", nullptr);
	

	truth.update_mass_properties();
	bezier.update_mass_properties();

	std::cout << truth.get_volume() << std::endl;
	std::cout << bezier.get_volume() << std::endl;
	// bezier.elevate_degree();

	bezier.populate_mass_properties_coefs_deterministics();

	bezier.update_mass_properties();
	std::cout << bezier.get_volume() << std::endl;



	return 0;
}