#include "Bezier.hpp"
#include "Facet.hpp"
#include "PC.hpp"
#include "ControlPoint.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeFitterBezier.hpp"




#include <chrono>
int main(){
	FrameGraph frame_graph;



	#ifdef __APPLE__



	ShapeModelImporter shape_io_guess("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test/apriori.obj", 1, true);
	PC pc("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/pc/source_transformed_poisson.obj");

	#elif __linux__
	
	ShapeModelImporter shape_io_guess("/home/bebe0705/libs/ASPEN_GUI_LESS/Apps/ShapeReconstruction/output/shape_model/apriori.obj", 1, true);
	PC pc("/home/bebe0705/libs/ASPEN_GUI_LESS/Apps/ShapeReconstruction/output/pc/source_transformed_poisson.obj");

	#endif

	ShapeModelTri a_priori_obj("", nullptr);

	shape_io_guess.load_obj_shape_model(&a_priori_obj);

	std::shared_ptr<ShapeModelBezier> a_priori_bezier = std::make_shared<ShapeModelBezier>(ShapeModelBezier(&a_priori_obj,"E", &frame_graph));

	// the shape is elevated to the prescribed degree
	unsigned int starting_degree = a_priori_bezier -> get_degree();
	a_priori_bezier -> elevate_degree();
	
	a_priori_bezier -> initialize_index_table();

	ShapeFitterBezier shape_fitter(a_priori_bezier.get(),&pc);

	shape_fitter.fit_shape_batch(5,0);



	return 0;
}