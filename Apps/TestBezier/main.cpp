#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "RigidBodyKinematics.hpp"



#include <chrono>


int main(){

	arma::arma_rng::set_seed(0);



	FrameGraph frame_graph;
	
	ShapeModelTri itokawa_true("", &frame_graph);
	ShapeModelBezier itokawa_bezier_fit("", &frame_graph);


	ShapeModelImporter shape_io_bezier("../../../resources/shape_models/cube_2.b", 1, true);
	ShapeModelImporter shape_io_true("../../../resources/shape_models/cube.obj", 1, true);
	
	// ShapeModelImporter shape_io_bezier("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/shape_model/test/fit_shape_aligned.b", 1, true);
	// ShapeModelImporter shape_io_true("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/shape_model/test/fit_shape_aligned.obj", 1, true);
	
	shape_io_bezier.load_bezier_shape_model(&itokawa_bezier_fit);
	shape_io_true.load_obj_shape_model(&itokawa_true);


	arma::vec x = {1,0,0};
	itokawa_bezier_fit.translate(x);
	itokawa_bezier_fit.update_mass_properties();



	itokawa_bezier_fit.save_to_obj("translated_cube.obj");

	itokawa_true.translate(x);
	itokawa_true.update_mass_properties();

	std::cout << "\nVolume: \n";

	std::cout << itokawa_bezier_fit.get_volume() << std::endl;
	std::cout << itokawa_true.get_volume() << std::endl;

	std::cout << "\nCenter of mass: \n";
	
	std::cout << itokawa_bezier_fit.get_center_of_mass() << std::endl;
	std::cout << itokawa_true.get_center_of_mass() << std::endl;

	std::cout << "\nInertia: \n";
	
	std::cout << itokawa_bezier_fit.get_inertia() << std::endl;
	std::cout << itokawa_true.get_inertia() << std::endl;

	arma::vec sigmas_sq = 0.01 * arma::ones<arma::vec>(3) * std::cbrt(itokawa_bezier_fit.get_volume());
	arma::mat P = arma::diagmat(sigmas_sq);

	itokawa_bezier_fit.shift_to_barycenter();
	itokawa_bezier_fit.update_mass_properties();
	std::cout << itokawa_bezier_fit.get_center_of_mass() << std::endl;






	itokawa_bezier_fit.get_control_point(0) -> set_covariance(P);
	itokawa_bezier_fit.get_control_point(1) -> set_covariance(P);
	itokawa_bezier_fit.get_control_point(2) -> set_covariance(P);
	// itokawa_bezier_fit.get_control_point(3) -> set_covariance(P);
	itokawa_bezier_fit.get_control_point(4) -> set_covariance(P);
	// itokawa_bezier_fit.get_control_point(5) -> set_covariance(P);
	itokawa_bezier_fit.get_control_point(6) -> set_covariance(P);
	itokawa_bezier_fit.get_control_point(7) -> set_covariance(P);

	itokawa_bezier_fit.compute_volume_sd();

	std::chrono::time_point<std::chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();
	itokawa_bezier_fit.compute_cm_cov();
	end = std::chrono::system_clock::now();


	std::chrono::duration<double> elapsed_seconds = end - start;

	std::cout << "\n Elapsed time computing the covariance : " << elapsed_seconds.count() << "s\n\n";


	double volume_mean = itokawa_bezier_fit.get_volume();
	double volume_sd = itokawa_bezier_fit.get_volume_sd();

	arma::vec cm_mean = itokawa_bezier_fit.get_center_of_mass();
	arma::mat cm_cov = itokawa_bezier_fit.get_cm_cov();
	// throw;


	std::cout << "\nVolume standard deviation: " << std::endl;
	std::cout << itokawa_bezier_fit.get_volume_sd() << std::endl;

	std::cout << "\nCenter of mass Covariance: " << std::endl;
	std::cout << cm_cov << std::endl;

	std::cout << "\nRunning Monte Carlo: " << std::endl;

	int N = 300000;


	arma::vec results_volume  = itokawa_bezier_fit.run_monte_carlo_volume(N);
	
	arma::mat results_cm  = itokawa_bezier_fit.run_monte_carlo_cm(N);

	arma::vec results_cm_mean = arma::mean(results_cm,1);
	arma::mat cov_cm_mc = arma::zeros(3,3);
	
	for (unsigned int i = 0; i < results_cm.n_cols; ++i){
		cov_cm_mc +=  (results_cm.col(i) - results_cm_mean) * (results_cm.col(i) - results_cm_mean).t();
	}

	cov_cm_mc *= 1./(results_cm.n_cols-1);




	std::cout << "######***** Volume *****######"<< std::endl;

	std::cout << "Mean volume from MC: " << arma::mean(results_volume) << std::endl;
	std::cout << "Mean volume predicted: " << volume_mean << std::endl << std::endl;

	std::cout << "SD volume from MC: " << arma::stddev(results_volume,1) << std::endl;
	std::cout << "SD volume predicted: " << volume_sd << std::endl << std::endl;
	std::cout << "Deviation : " << (arma::stddev(results_volume,1) - volume_sd)/volume_sd * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** CM *****######"<< std::endl;

	std::cout << "Mean CM from MC: " << results_cm_mean.t() << std::endl;
	std::cout << "Mean CM predicted: " << cm_mean.t() << std::endl;

	std::cout << "COV CM from MC: " << std::endl << cov_cm_mc << std::endl;
	std::cout << "COV CM predicted: " << std::endl << cm_cov << std::endl << std::endl;


	std::cout << "Deviation : " << std::endl << (cov_cm_mc - cm_cov)/cov_cm_mc * 100 << " %" << std::endl << std::endl;



	return 0;
}