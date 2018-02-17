#include "ControlPoint.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeBuilder.hpp"

#include "ShapeFitterBezier.hpp"
#include "PC.hpp"
#include "Lidar.hpp"




#include <chrono>


int main(){

	arma::arma_rng::set_seed(0);
	// True shape model
	ShapeModelImporter shape_io_bezier("../input/true_patch.b", 1, true);
	ShapeModelImporter shape_perturbed_bezier("../input/true_patch.b", 1, true);

	FrameGraph frame_graph;
	frame_graph.add_frame("N");
	frame_graph.add_frame("B");
	frame_graph.add_frame("L");
	frame_graph.add_frame("E");


	frame_graph.add_transform("N", "B");
	frame_graph.add_transform("N", "E");
	
	frame_graph.add_transform("N", "L");

	ShapeModelBezier nominal_patch("B", &frame_graph);
	ShapeModelBezier perturbed_patch("E", &frame_graph);


	shape_io_bezier.load_bezier_shape_model(&nominal_patch);
	shape_perturbed_bezier.load_bezier_shape_model(&perturbed_patch);



	for (unsigned int i = 0; i < perturbed_patch . get_control_points() -> size(); ++i){
		auto p = perturbed_patch . get_control_points() -> at(i);
		arma::vec dp = 0.3 * arma::randn<arma::vec>(3);
		p -> set_coordinates(p -> get_coordinates() + dp);
	}
	perturbed_patch.elevate_degree();

	

	nominal_patch.save_both("../input/true_patch");

	perturbed_patch.save_both("../output/perturbed");


	Lidar lidar(&frame_graph,
		"L",
		20,
		20 ,
		128,
		128,
		1e-1,
		1,
		1e-2,
		0);


	// Setting the Lidar frame to its new state
	arma::vec lidar_pos = {0,0,5};
	
	arma::vec target_pos = {0,0,0};
	arma::vec mrp_BN = {0.0,0.00,0};


	arma::vec l1 = arma::normalise(dynamic_cast<Bezier *>(nominal_patch. get_elements() -> at(0).get()) -> get_center() - lidar_pos);
	arma::vec l2 = arma::normalise(arma::cross(arma::randu<arma::vec>(3),l1));
	arma::vec l3 = arma::cross(l1,l2);

	arma::mat LN(3,3);
	LN.row(0) = l1.t();
	LN.row(1) = l2.t();
	LN.row(2) = l3.t();

	arma::vec mrp_LN = RBK::dcm_to_mrp(LN);


	frame_graph.get_frame("L") -> set_origin_from_parent(lidar_pos);
	frame_graph.get_frame("L") -> set_mrp_from_parent(mrp_LN);

	frame_graph.get_frame("E") -> set_origin_from_parent(target_pos);
	frame_graph.get_frame("E") -> set_mrp_from_parent(mrp_BN);

	frame_graph.get_frame("B") -> set_origin_from_parent(target_pos);
	frame_graph.get_frame("B") -> set_mrp_from_parent(mrp_BN);

	lidar.send_flash(&nominal_patch,true);

	PC pc_true(lidar.get_focal_plane());
	pc_true.save("../output/true_pc_before_transform.obj");
	pc_true.transform(LN.t(),lidar_pos);
	pc_true.save("../output/true_pc.obj");

	ShapeFitterBezier shape_fitter(&perturbed_patch,
		&pc_true);

	auto footpoints = shape_fitter.fit_shape_KF(0,
		20,
		1e-5,
		1,
		l1);
	perturbed_patch.save_both("../output/fit");

    // The patch covariance is learned
	arma::mat P_X(18,18);
	Bezier * patch = dynamic_cast<Bezier *>(perturbed_patch. get_elements() -> at(0).get());
	patch -> train_patch_covariance(P_X,footpoints);


	std::cout << "Trained covariance: " << std::endl;
	std::cout << P_X  << std::endl;

	// The error is measured between each point in the true noisy pc
	// and the pc obtained over the fit patch

	arma::mat results(4,footpoints.size());


	for (unsigned int i = 0; i < footpoints.size(); ++i){
		Footpoint footpoint = footpoints[i];
		arma::vec dir = footpoint.n;
		arma::mat P = patch -> covariance_surface_point(footpoint.u,footpoint.v,dir,P_X);

		double sd = std::sqrt(arma::dot(dir,P * dir));
		double residual = arma::dot(dir,footpoint.Ptilde - footpoint.Pbar );

		results(0,i) = sd;
		results(1,i) = residual;
		results(2,i) = footpoint.u;
		results(3,i) = footpoint.v;

	}


	results.save("../output/results.txt",arma::raw_ascii);

	

	return 0;

}