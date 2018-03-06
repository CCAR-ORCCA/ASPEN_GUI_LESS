#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"


#include <chrono>


int main(){

	FrameGraph frame_graph;
	
	ShapeModelTri itokawa_true("", &frame_graph);
	ShapeModelBezier itokawa_bezier_fit("", &frame_graph);

	ShapeModelImporter shape_io_bezier("../../ShapeReconstruction/output/shape_model/test/fit_source_400.b", 1, true);
	ShapeModelImporter shape_io_true("../../../resources/shape_models/itokawa_64_scaled_aligned.obj", 1, true);
	
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

	// The shape model is shifted so as to have its coordinates
		// expressed in its barycentric frame
	itokawa_bezier_fit . shift_to_barycenter();

	itokawa_bezier_fit . update_mass_properties();

		// The shape model is then rotated so as to be oriented
		// with respect to its principal axes
	itokawa_bezier_fit . align_with_principal_axes();


	itokawa_bezier_fit.save_both("aligned");



	return 0;
}