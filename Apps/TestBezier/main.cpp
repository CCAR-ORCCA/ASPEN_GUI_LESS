#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "RigidBodyKinematics.hpp"



#include <chrono>


int main(){




	FrameGraph frame_graph;
	
	ShapeModelTri tri_shape("", &frame_graph);

	ShapeModelImporter shape_io_true("../../../resources/shape_models/itokawa_8.obj", 1, false);

	shape_io_true.load_obj_shape_model(&tri_shape);
	arma::vec mrp = {0.1,0.2,-0.2};
	arma::mat M = RBK::mrp_to_dcm(mrp);
	tri_shape.rotate(M);
	tri_shape.update_mass_properties();	

	

	ShapeModelBezier bezier_shape(&tri_shape,"", &frame_graph);


	

	std::cout << "\nVolume: \n";

	std::cout << bezier_shape.get_volume() << std::endl;
	std::cout << tri_shape.get_volume() << std::endl;

	std::cout << "\nCenter of mass: \n";
	
	std::cout << bezier_shape.get_center_of_mass() << std::endl;
	std::cout << tri_shape.get_center_of_mass() << std::endl;

	std::cout << "\nInertia: \n";
	
	std::cout << bezier_shape.get_inertia() << std::endl;
	std::cout << tri_shape.get_inertia() << std::endl;

	arma::vec sigmas_sq = (0.00300 * 0.00300) * arma::ones<arma::vec>(3);
	
	// arma::vec sigmas_sq = 0.01 * arma::ones<arma::vec>(3) * std::cbrt(bezier_shape.get_volume());
	std::cout << "SD on point coordinates : " << std::sqrt(sigmas_sq(0)) << std::endl;

	arma::mat P = arma::diagmat(sigmas_sq);

	bezier_shape.shift_to_barycenter();
	bezier_shape.update_mass_properties();


	for (unsigned int i = 0; i < bezier_shape.get_NControlPoints(); ++i){
		arma::vec x = arma::randu<arma::vec>(1);
		double factor = 5 * x(0);
		bezier_shape.get_control_point(i) -> set_covariance(factor * P );
	}

	bezier_shape.compute_volume_sd();

	std::chrono::time_point<std::chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();
	bezier_shape.compute_cm_cov();
	end = std::chrono::system_clock::now();
	bezier_shape.compute_inertia_statistics();



	std::chrono::duration<double> elapsed_seconds = end - start;

	std::cout << "\n Elapsed time computing the covariance : " << elapsed_seconds.count() << "s\n\n";

	double volume_mean = bezier_shape.get_volume();
	double volume_sd = bezier_shape.get_volume_sd();

	arma::vec cm_mean = bezier_shape.get_center_of_mass();
	arma::mat cm_cov = bezier_shape.get_cm_cov();

	arma::mat inertia_cov = bezier_shape.get_inertia_cov();
	arma::mat moments_cov = bezier_shape.get_P_moments();
	// throw;


	std::cout << "\nVolume standard deviation: " << std::endl;
	std::cout << bezier_shape.get_volume_sd() << std::endl;

	std::cout << "\nCenter of mass Covariance: " << std::endl;
	std::cout << cm_cov << std::endl;

	std::cout << "\nRunning Monte Carlo: " << std::endl;

	int N = 50000;

	arma::vec results_volume;
	arma::mat results_cm,results_inertia,results_moments;
	bezier_shape.run_monte_carlo(N,results_volume,results_cm,results_inertia,results_moments);
	

	arma::vec results_cm_mean = arma::mean(results_cm,1);
	arma::vec results_inertia_mean = arma::mean(results_inertia,1);
	arma::vec results_moments_mean = arma::mean(results_moments,1);

	arma::mat cov_cm_mc = arma::zeros(3,3);
	arma::mat cov_inertia_mc = arma::zeros(6,6);
	arma::mat cov_moments_mc = arma::zeros(4,4);


	
	for (unsigned int i = 0; i < results_cm.n_cols; ++i){
		cov_cm_mc +=  (results_cm.col(i) - results_cm_mean) * (results_cm.col(i) - results_cm_mean).t();
		cov_inertia_mc +=  (results_inertia.col(i) - results_inertia_mean) * (results_inertia.col(i) - results_inertia_mean).t();
		cov_moments_mc +=  (results_moments.col(i) - results_moments_mean) * (results_moments.col(i) - results_moments_mean).t();


	}

	cov_cm_mc *= 1./(results_cm.n_cols-1);
	cov_inertia_mc *= 1./(results_inertia.n_cols-1);
	cov_moments_mc *= 1./(results_moments.n_cols-1);


	std::cout << "######***** Volume *****######"<< std::endl;

	
	std::cout << "SD volume from MC: " << arma::stddev(results_volume,1) << std::endl;
	std::cout << "SD volume predicted: " << volume_sd << std::endl << std::endl;
	std::cout << "Deviation : " << (arma::stddev(results_volume,1) - volume_sd)/volume_sd * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** CM *****######"<< std::endl;


	std::cout << "COV CM from MC: " << std::endl << cov_cm_mc << std::endl;
	std::cout << "COV CM predicted: " << std::endl << cm_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_cm_mc - cm_cov)/cov_cm_mc * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** Inertia *****######"<< std::endl;

	std::cout << "Mean inertia : " << results_inertia_mean.t() << std::endl;
	std::cout << "COV Inertia parametrization from MC: " << std::endl << cov_inertia_mc << std::endl;
	std::cout << "COV Inertia parametrization predicted: " << std::endl << inertia_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_inertia_mc - inertia_cov)/cov_inertia_mc * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** Moments *****######"<< std::endl;

	std::cout << "Mean moments : " << results_moments_mean.t() << std::endl;
	std::cout << "Inertia moments covariance from MC: " << std::endl << cov_moments_mc << std::endl;
	std::cout << "Inertia moments covariance predicted: " << std::endl << moments_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_moments_mc - moments_cov)/cov_moments_mc * 100 << " %" << std::endl << std::endl;



	return 0;
}