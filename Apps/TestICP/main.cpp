
#include "PC.hpp"
#include "ICP.hpp"
#include <armadillo>
#include "boost/progress.hpp"
#include <ShapeModelImporter.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>


int main() {


	// ShapeModelTri true_shape_model("", nullptr);
	// ShapeModelImporter true_shape_model_io(
	// 	"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_64_scaled_aligned.obj", 1, true);


	// true_shape_model_io.load_obj_shape_model(&true_shape_model);

	// arma::mat points = true_shape_model.random_sampling(10);
	// arma::vec los = {1,0,0};
	// PC pc_truth(los,points);
	// pc_truth.save("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/pc/true_pc_prealigned.obj");


	std::shared_ptr<PC> pc_truth = std::make_shared<PC>(PC("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/pc/true_pc_prealigned.obj"));
	std::shared_ptr<PC> pc_fit = std::make_shared<PC>(PC("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/pc/source_transformed_poisson.obj"));

	
	ICP icp_pc(
		pc_fit, 
		pc_truth, 
		arma::eye<arma::mat>(3,3), 
		arma::zeros<arma::vec>(3),
		true);


	arma::mat M = icp_pc.get_M();
	arma::vec X = icp_pc.get_X();
	pc_truth ->  save("../registered_truth.obj",M,X);



	// arma::mat source_pc_mat;
	// arma::mat destination_pc_mat;

	// source_pc_mat.load("../source.txt");
	// destination_pc_mat.load("../destination.txt");

	// arma::inplace_trans(source_pc_mat);
	// arma::inplace_trans(destination_pc_mat);


	// arma::vec u = {1,0,0};

	// std::shared_ptr<PC> source_pc = std::make_shared<PC>(PC(u,source_pc_mat));
	// std::shared_ptr<PC> destination_pc = std::make_shared<PC>(PC(u,destination_pc_mat));
	// source_pc -> save("../original_source.obj");
	// destination_pc -> save("../destination.obj");


	// unsigned int N_iter = 100;

	
	// ICP icp_pc(
	// 	destination_pc, 
	// 	source_pc, 
	// 	arma::eye<arma::mat>(3,3), 
	// 	arma::zeros<arma::vec>(3),
	// 	true);

	// arma::mat M = icp_pc.get_M();
	// arma::vec X = icp_pc.get_X();


	// source_pc -> save("../registered_source.obj",M,X);

	// std::cout << "DCM: \n";
	// std::cout << M << std::endl;
	// std::cout << "X: \n";

	// std::cout << X << std::endl;




	return 0;
}












