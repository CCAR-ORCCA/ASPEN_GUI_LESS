#include "ControlPoint.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeFitterBezier.hpp"
#include "PC.hpp"



#include <chrono>


int main(){

	FrameGraph frame_graph;

	std::string asteroid = "bennu";
	std::string guess = "faceted_sphere";

	// ShapeModelImporter shape_io(asteroid + ".obj", 1, true);
	// ShapeModelTri true_shape("",&frame_graph);
	// shape_io.load_obj_shape_model(&true_shape);

	// ShapeModelBezier bezier(&true_shape,"",&frame_graph);

	// bezier.save(asteroid + ".b");

	ShapeModelImporter shape_io(asteroid + ".b", 1, true);
	ShapeModelBezier true_shape("",&frame_graph);
	shape_io.load_bezier_shape_model(&true_shape);

	ShapeModelImporter fit_shape_io(guess + ".b", 0.3, true);
	ShapeModelBezier fit_shape("",&frame_graph);
	fit_shape_io.load_bezier_shape_model(&fit_shape);

	fit_shape.elevate_degree();
	fit_shape.elevate_degree();


	fit_shape.save_to_obj("shrunk_asteroid.obj");

	unsigned int degree = fit_shape.get_degree();

	// A point cloud is collected over the surface of the object
	unsigned int N_points = 10;

	arma::mat points(3,N_points * true_shape.get_NElements());

	for (unsigned int e = 0; e < true_shape.get_NElements(); ++e){
		for (unsigned int i = 0; i < N_points; ++i){
			arma::vec rand = arma::randu<arma::vec>(2);
			double u = rand(0);
			double v = (1 - u) * rand(1);
			points.col(e * N_points + i) = dynamic_cast<Bezier *>(true_shape.get_elements() -> at(e).get()) -> evaluate(u,v) + 0.003 * arma::randn<arma::vec>(3);
		}
	}

	PC::save(points,"pc.obj");


	arma::vec u = {1,0,0};
	PC pc(u,points);

	// The shape is fit
	unsigned int N_iter = 1;
	ShapeFitterBezier shape_fitter(&fit_shape,&pc);

	shape_fitter.fit_shape_batch(N_iter,1e-5,arma::eye<arma::mat>(3,3), arma::zeros<arma::vec>(3));

	fit_shape.save_to_obj(guess + "_fit_degree_" + std::to_string(degree) +"_iter_" + std::to_string(N_iter) + ".b");


	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();

	fit_shape.save_to_obj(guess + "_fit_degree_"+std::to_string(degree) +"_iter_" + std::to_string(N_iter) + ".obj");




	return 0;

}