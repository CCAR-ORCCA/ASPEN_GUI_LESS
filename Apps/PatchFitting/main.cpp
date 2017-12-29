#include "ControlPoint.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeFitterBezier.hpp"
#include "PC.hpp"



#include <chrono>


int main(){

	FrameGraph frame_graph;

	std::string asteroid = "itokawa_hr";
	std::string guess = "faceted_sphere";

	// ShapeModelImporter shape_io(asteroid + ".obj", 1, true);
	// ShapeModelTri true_shape("",&frame_graph);
	// shape_io.load_obj_shape_model(&true_shape);

	// ShapeModelBezier bezier(&true_shape,"",&frame_graph);

	// bezier.save(asteroid + ".b");

	// ShapeModelImporter shape_io(asteroid + ".b", 1, true);
	// ShapeModelBezier true_shape("",&frame_graph);
	// shape_io.load_bezier_shape_model(&true_shape);

	ShapeModelImporter fit_shape_io(guess + ".b", 500, true);
	
	ShapeModelBezier fit_shape("",&frame_graph);
	fit_shape_io.load_bezier_shape_model(&fit_shape);


	fit_shape.elevate_degree();
	fit_shape.elevate_degree();

	unsigned int degree = fit_shape.get_degree();

	arma::mat points;
	points.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/paperGNSki/data/sampled_pc_itokawa_scaled_aligned.txt");
	if (points.n_rows > points.n_cols)
		arma::inplace_trans(points);

	// Compute the latitude/longitude/radius of a number of a selected number of points in the point cloud

	unsigned int N_sample = points.n_cols;
	arma::ivec indices = arma::randi( N_sample, arma::distr_param(0,points.n_cols - 1) ) ;

	arma::mat long_lat_rad = arma::zeros<arma::mat>(N_sample,3);

	#pragma omp parallel for
	for (unsigned int i = 0; i < indices.n_rows; ++i){
		arma::vec point = points.col(indices(i));
		double longitude = std::atan2(point(1),point(0));
		double latitude = std::atan2(point(2),arma::norm(point.rows(0,1)));
		double radius = arma::norm(point);

		long_lat_rad(i,0) = longitude;
		long_lat_rad(i,1) = latitude;
		long_lat_rad(i,2) = radius;

	}

	// The control points to the a-priori are relocated closer to the selected points
	for (unsigned int i = 0 ; i < fit_shape.get_NControlPoints(); ++i){

		auto point_ptr = fit_shape.get_control_point(i);
		auto point = point_ptr -> get_coordinates();
		arma::rowvec long_lat(2);
		long_lat(0) = std::atan2(point(1),point(0));
		long_lat(1) = std::atan2(point(2),arma::norm(point.rows(0,1)));

		double min_distance = arma::norm(long_lat_rad.row(0).cols(0,1) - long_lat);
		unsigned int best_e = 0;

		for (unsigned int e = 0; e < indices.n_rows; ++e){
			double distance = arma::norm(long_lat_rad.row(e).cols(0,1) - long_lat);
			if (distance < min_distance){
				min_distance = distance;
				best_e = e;
			}

		}

		point_ptr -> set_coordinates( long_lat_rad(best_e,2) * arma::normalise(point) );

	}

	fit_shape.construct_kd_tree_control_points();

	// // The point cloud is created
	arma::vec u = {1,0,0};
	PC pc(u,points);


	fit_shape.save_to_obj("a_priori_" + asteroid + "_degree_" +std::to_string(degree) + ".obj");
	fit_shape.save("a_priori_" + asteroid + "_degree_" +std::to_string(degree) + ".b");

	// // The shape is fit
	unsigned int N_iter = 10;

	ShapeFitterBezier shape_fitter(&fit_shape,&pc);

	shape_fitter.fit_shape_batch(N_iter,1e-5,arma::eye<arma::mat>(3,3), arma::zeros<arma::vec>(3));

	fit_shape.save(asteroid + "_fit_degree_" + std::to_string(degree) +"_iter_" + std::to_string(N_iter) + ".b");

	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();
	fit_shape.elevate_degree();

	fit_shape.save_to_obj(asteroid + "_fit_degree_"+std::to_string(degree) +"_iter_" + std::to_string(N_iter) + ".obj");




	return 0;

}