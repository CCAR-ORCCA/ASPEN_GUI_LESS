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

	ShapeModelTri truth("", nullptr);
	ShapeModelBezier bezier("", nullptr);




	
	ShapeModelImporter shape_io_truth("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/cube.obj", 1, true);
	shape_io_truth.load_obj_shape_model(&truth);	




	ShapeModelImporter shape_io_bezier("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/cube.b", 1, true);


	shape_io_bezier.load_bezier_shape_model(&bezier);

	truth.construct_kd_tree_shape();


	// The shape error is computed here

	int facet_index = 4;
	Bezier * patch = static_cast<Bezier *>( bezier.get_elements() -> at(facet_index).get() );
	arma::vec normal = patch -> get_normal_coordinates(1./3, 1./3);
	arma::vec center = patch -> evaluate(1./3,1./3) - 0.01 * normal;

	std::cout << "origin: " << center.t() << std::endl;
	std::cout << "direction: " << normal.t() << std::endl;


	Ray ray_n(center,normal);
	std::cout << "##################### forward\n";

	truth.ray_trace(&ray_n);

	Ray ray_mn(center,-normal);
	std::cout << "##################### backward\n";

	truth.ray_trace(&ray_mn);


	if (ray_n.get_true_range() < ray_mn.get_true_range()){
			// shape_error_results.push_back({sd,ray_n.get_true_range()});

	}
	else if (ray_n.get_true_range() > ray_mn.get_true_range()){
			// shape_error_results.push_back({sd,ray_mn.get_true_range()});

	}
	else{
		std::cout << "didn't hit\n";
	}

	
















	return 0;
}