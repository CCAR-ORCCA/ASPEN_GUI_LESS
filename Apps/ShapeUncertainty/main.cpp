#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "RigidBodyKinematics.hpp"
#include "json.hpp"



#include <chrono>

#include <armadillo>


int main(){

	std::ifstream i("input_file.json");
	nlohmann::json input_data;
	i >> input_data;

	std::string path_shape = input_data["PATH_SHAPE"];
	double correlation_distance =  input_data["CORRELATION_DISTANCE"];

	double error_standard_dev  = input_data["ERROR_STANDARD_DEV"];
	int N_monte_carlo = input_data["N_MONTE_CARLO"];
	std::string output_dir = input_data["dir"];

	std::cout << "Path to shape: " << path_shape << std::endl;
	std::cout << "Standard deviation on point coordinates (km) : " << error_standard_dev << std::endl;
	std::cout << "Correlation distance (km) : " << correlation_distance << std::endl;
	std::cout << "Monte Carlo Draws : " << N_monte_carlo << std::endl;


	FrameGraph frame_graph;
	ShapeModelTri tri_shape("", &frame_graph);

	ShapeModelImporter::load_obj_shape_model(path_shape, 1, true,tri_shape);
	ShapeModelBezier bezier_shape(tri_shape,"", &frame_graph);

	std::cout << "\nVolume (km^3): \n";
	std::cout << tri_shape.get_volume() << " / " << bezier_shape.get_volume() << std::endl;

	std::cout << "\nCenter-of-mass (km): \n";
	std::cout << tri_shape.get_center_of_mass().t() << " / " << bezier_shape.get_center_of_mass().t() << std::endl;

	std::cout << "\nVolume error: (%) \n";
	std::cout << std::abs(bezier_shape.get_volume() - tri_shape.get_volume())/bezier_shape.get_volume() * 100 << std::endl;

	std::cout << "\nCenter of mass error: (absolute) \n";
	std::cout << arma::abs(bezier_shape.get_center_of_mass() - tri_shape.get_center_of_mass())<< std::endl;

	std::cout << "\nCenter of mass error: (%) \n";
	std::cout << arma::abs(bezier_shape.get_center_of_mass() - tri_shape.get_center_of_mass())/arma::abs(bezier_shape.get_center_of_mass()) * 100<< std::endl;

	std::cout << "\nInertia error:  (absolute) \n";
	std::cout << arma::abs(bezier_shape.get_inertia() - tri_shape.get_inertia()) << std::endl;

	std::cout << "\nInertia error:  (%) \n";
	std::cout << arma::abs(bezier_shape.get_inertia() - tri_shape.get_inertia())/arma::abs(bezier_shape.get_inertia()) * 100 << std::endl;
	
	bezier_shape.compute_point_covariances(std::pow(error_standard_dev,2),correlation_distance);
	bezier_shape.compute_shape_covariance_sqrt();

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	bezier_shape.compute_all_statistics();
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "\n Elapsed time computing all covariances : " << elapsed_seconds.count() << "s\n\n";

	

	double volume_sd = bezier_shape.get_volume_sd();
	auto cm_mean = bezier_shape.get_center_of_mass();
	auto cm_cov = bezier_shape.get_cm_cov();
	auto inertia_cov = bezier_shape.get_inertia_cov();
	auto moments_cov = bezier_shape.get_P_moments();
	auto mrp_cov = bezier_shape.get_mrp_cov();
	auto Y_cov = bezier_shape.get_P_Y();
	auto lambda_I_cov = bezier_shape.get_P_lambda_I(2);
	auto eigenvectors_cov = bezier_shape.get_P_eigenvectors();
	auto Evectors_cov = bezier_shape.get_P_Evectors();
	auto MI_cov = bezier_shape.get_P_MI();
	auto dims_cov = bezier_shape.get_P_dims();

	// Saving shape results

	arma::vec volume_vec(1);
	arma::vec eigenvectors(9);
	arma::vec Y(4);
	arma::vec MI(7);
	arma::vec dims(3);
	arma::vec moments(4);
	arma::vec I(6);
	arma::vec mrp(3);
	arma::vec center_of_mass(3);
	arma::vec Evectors(9);


	double volume = bezier_shape.get_volume();
	volume_vec(0) = volume;
	center_of_mass = bezier_shape.get_center_of_mass();
	arma::mat::fixed<3,3> inertia  = bezier_shape.get_inertia();

	arma::mat I_C = inertia - volume * RBK::tilde(center_of_mass) * RBK::tilde(center_of_mass).t() ;
	I = {I_C(0,0),I_C(1,1),I_C(2,2),I_C(0,1),I_C(0,2),I_C(1,2)};

	arma::vec eig_val = arma::eig_sym(I_C);
	arma::mat eig_vec =  ShapeModelBezier::get_principal_axes_stable(I_C);
	moments.rows(0,2) = eig_val;
	moments(3) = volume;

	
	mrp = RBK::dcm_to_mrp(eig_vec);

	eigenvectors.rows(0,2) = eig_vec.col(0);
	eigenvectors.rows(3,5) = eig_vec.col(1);
	eigenvectors.rows(6,8) = eig_vec.col(2);
	

	Evectors = ShapeModelBezier::get_E_vectors(I_C);
	Y = ShapeModelBezier::get_Y(volume,I_C);
	MI.rows(0,5) = I;
	MI(6) = volume;
	dims = ShapeModelBezier::get_dims(volume,I_C);


	volume_vec.save(output_dir + "/volume.txt",arma::raw_ascii);
	eigenvectors.save(output_dir + "/eigenvectors.txt",arma::raw_ascii);
	Y.save(output_dir + "/Y.txt",arma::raw_ascii);
	MI.save(output_dir + "/MI.txt",arma::raw_ascii);
	dims.save(output_dir + "/dims.txt",arma::raw_ascii);
	moments.save(output_dir + "/moments.txt",arma::raw_ascii);
	I.save(output_dir + "/I.txt",arma::raw_ascii);
	mrp.save(output_dir + "/mrp.txt",arma::raw_ascii);
	center_of_mass.save(output_dir + "/center_of_mass.txt",arma::raw_ascii);
	Evectors.save(output_dir + "/Evectors.txt",arma::raw_ascii);


	arma::vec sd_volume_predicted = {volume_sd};
	
	sd_volume_predicted.save(output_dir + "/sd_volume.txt",arma::raw_ascii);
	cm_cov.save(output_dir + "/cm_cov.txt",arma::raw_ascii);
	inertia_cov.save(output_dir + "/inertia_cov.txt",arma::raw_ascii);
	moments_cov.save(output_dir + "/moments_cov.txt",arma::raw_ascii);
	dims_cov.save(output_dir + "/dims_cov.txt",arma::raw_ascii);
	mrp_cov.save(output_dir + "/mrp_cov.txt",arma::raw_ascii);







	std::cout << "\nRunning Monte Carlo: " << std::endl;

	int N = N_monte_carlo;

	arma::vec results_volume;
	arma::mat results_cm,results_inertia,results_moments,
	results_mrp,results_lambda_I,results_eigenvectors,results_Evectors,results_Y,results_MI,
	results_dims;
	
	start = std::chrono::system_clock::now();

	bezier_shape.run_monte_carlo(N,
		results_volume,
		results_cm,
		results_inertia,
		results_moments,
		results_mrp,
		results_lambda_I,
		results_eigenvectors,
		results_Evectors,
		results_Y,
		results_MI,
		results_dims,
		output_dir);

	end = std::chrono::system_clock::now();

	elapsed_seconds = end - start;
	
	std::cout << "\nDone running Monte Carlo in " << elapsed_seconds.count() << " s\n";


	
	arma::vec results_cm_mean = arma::mean(results_cm,1);
	arma::vec results_inertia_mean = arma::mean(results_inertia,1);
	arma::vec results_moments_mean = arma::mean(results_moments,1);
	arma::vec results_mrp_mean = arma::mean(results_mrp,1);
	arma::vec results_lambda_I_mean = arma::mean(results_lambda_I,1);
	arma::vec results_eigenvectors_mean = arma::mean(results_eigenvectors,1);
	arma::vec results_Evectors_mean = arma::mean(results_Evectors,1);
	arma::vec results_Y_mean = arma::mean(results_Y,1);
	arma::vec results_MI_mean = arma::mean(results_MI,1);
	arma::vec results_dims_mean = arma::mean(results_dims,1);



	arma::mat cov_cm_mc = arma::zeros(3,3);
	arma::mat cov_inertia_mc = arma::zeros(6,6);
	arma::mat cov_moments_mc = arma::zeros(4,4);
	arma::mat cov_mrp_mc = arma::zeros(3,3);
	arma::vec cov_lambda_I_mc = arma::zeros<arma::vec>(6);
	arma::mat cov_eigenvectors_mc = arma::zeros<arma::mat>(9,9);
	arma::mat cov_Evectors_mc = arma::zeros<arma::mat>(9,9);
	arma::mat cov_Y_mc = arma::zeros<arma::mat>(4,4);
	arma::vec cov_MI_mc = arma::zeros<arma::vec>(6);
	arma::mat cov_dims_mc = arma::zeros<arma::mat>(3,3);


	
	for (unsigned int i = 0; i < results_cm.n_cols; ++i){

		cov_cm_mc +=  (results_cm.col(i) - results_cm_mean) * (results_cm.col(i) - results_cm_mean).t();
		cov_inertia_mc +=  (results_inertia.col(i) - results_inertia_mean) * (results_inertia.col(i) - results_inertia_mean).t();
		cov_moments_mc +=  (results_moments.col(i) - results_moments_mean) * (results_moments.col(i) - results_moments_mean).t();
		cov_mrp_mc +=  (results_mrp.col(i) - results_mrp_mean) * (results_mrp.col(i) - results_mrp_mean).t();
		cov_lambda_I_mc += (results_lambda_I.col(i).rows(0,5) - results_lambda_I_mean.rows(0,5)) * (results_lambda_I.col(i)(6) - results_lambda_I_mean(6));

		cov_eigenvectors_mc +=  (results_eigenvectors.col(i) - results_eigenvectors_mean) * (results_eigenvectors.col(i) - results_eigenvectors_mean).t();
		cov_Evectors_mc +=  (results_Evectors.col(i) - results_Evectors_mean) * (results_Evectors.col(i) - results_Evectors_mean).t();
		cov_Y_mc +=  (results_Y.col(i) - results_Y_mean) * (results_Y.col(i) - results_Y_mean).t();

		cov_MI_mc +=  (results_MI.col(i).rows(0,5) - results_MI_mean.rows(0,5)) * (results_MI.col(i)(6) - results_MI_mean(6));
		cov_dims_mc +=  (results_dims.col(i) - results_dims_mean) * (results_dims.col(i) - results_dims_mean).t();

	}

	cov_cm_mc *= 1./(results_cm.n_cols-1);
	cov_inertia_mc *= 1./(results_inertia.n_cols-1);
	cov_moments_mc *= 1./(results_moments.n_cols-1);
	cov_mrp_mc *= 1./(results_mrp.n_cols-1);
	cov_lambda_I_mc *= 1./(results_lambda_I.n_cols-1);
	cov_eigenvectors_mc *= 1./(results_eigenvectors.n_cols-1);
	cov_Evectors_mc *= 1./(results_Evectors.n_cols-1);
	cov_Y_mc *= 1./(results_Y.n_cols-1);
	cov_MI_mc *= 1./(results_MI.n_cols-1);
	cov_dims_mc *= 1./(results_dims.n_cols-1);


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

	std::cout << "######***** MI *****######"<< std::endl;

	std::cout << "Mean MI : " << results_MI_mean.t() << std::endl;
	std::cout << "Inertia MI covariance from MC: " << std::endl << cov_MI_mc << std::endl;
	std::cout << "Inertia MI covariance predicted: " << std::endl << MI_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_MI_mc - MI_cov)/cov_MI_mc * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** Y *****######"<< std::endl;

	std::cout << "Mean Y : " << results_Y_mean.t() << std::endl;
	std::cout << "Inertia Y covariance from MC: " << std::endl << cov_Y_mc << std::endl;
	std::cout << "Inertia Y covariance predicted: " << std::endl << Y_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_Y_mc - Y_cov)/cov_Y_mc * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** Moments *****######"<< std::endl;

	std::cout << "Mean moments : " << results_moments_mean.t() << std::endl;
	std::cout << "Inertia moments covariance from MC: " << std::endl << cov_moments_mc << std::endl;
	std::cout << "Inertia moments covariance predicted: " << std::endl << moments_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_moments_mc - moments_cov)/cov_moments_mc * 100 << " %" << std::endl << std::endl;

	std::cout << "######***** Principal dimensions *****######"<< std::endl;

	std::cout << "Mean dimensions : " << results_dims_mean.t() << std::endl;
	std::cout << "Inertia dimensions covariance from MC: " << std::endl << cov_dims_mc << std::endl;
	std::cout << "Inertia dimensions covariance predicted: " << std::endl << dims_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_dims_mc - dims_cov)/cov_dims_mc * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** Lambda_I *****######"<< std::endl;

	std::cout << "Mean Lambda_I : " << results_lambda_I_mean.t() << std::endl;
	std::cout << "Lambda_I covariance from MC: " << std::endl << cov_lambda_I_mc << std::endl;
	std::cout << "Lambda_I covariance predicted: " << std::endl << lambda_I_cov.t() << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_lambda_I_mc - lambda_I_cov.t())/cov_lambda_I_mc * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** Evectors *****######"<< std::endl;

	std::cout << "Mean Evectors : " << results_Evectors_mean.t() << std::endl;
	std::cout << "Evectors covariance from MC: " << std::endl << cov_Evectors_mc << std::endl;
	std::cout << "Evectors covariance predicted: " << std::endl << Evectors_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_Evectors_mc - Evectors_cov)/cov_Evectors_mc * 100 << " %" << std::endl << std::endl;

	std::cout << "######***** Eigenvectors *****######"<< std::endl;

	std::cout << "Mean Eigenvectors : " << results_eigenvectors_mean.t() << std::endl;
	std::cout << "Eigenvectors covariance from MC: " << std::endl << cov_eigenvectors_mc << std::endl;
	std::cout << "Eigenvectors covariance predicted: " << std::endl << eigenvectors_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_eigenvectors_mc - eigenvectors_cov)/cov_eigenvectors_mc * 100 << " %" << std::endl << std::endl;


	std::cout << "######***** MRP *****######"<< std::endl;

	std::cout << "Mean MRP : " << results_mrp_mean.t() << std::endl;
	std::cout << "MRP covariance from MC: " << std::endl << cov_mrp_mc << std::endl;
	std::cout << "MRP covariance predicted: " << std::endl << mrp_cov << std::endl << std::endl;
	std::cout << "Deviation : " << std::endl << (cov_mrp_mc - mrp_cov)/cov_mrp_mc * 100 << " %" << std::endl << std::endl;



	
	arma::vec sd_volume_mc = {arma::stddev(results_volume,1)};
	arma::vec results_volume_mean = {arma::mean(results_volume)};


	sd_volume_mc.save(output_dir + "/sd_volume_mc.txt",arma::raw_ascii);
	results_volume.save(output_dir +"/volume_spread.txt",arma::raw_ascii);
	results_volume_mean.save(output_dir +"/volume_mean.txt",arma::raw_ascii);


	cov_cm_mc.save(output_dir + "/cov_cm_mc.txt",arma::raw_ascii);
	results_cm.save(output_dir + "/cm_spread.txt",arma::raw_ascii);
	results_cm_mean.save(output_dir + "/cm_mean.txt",arma::raw_ascii);

	

	cov_inertia_mc.save(output_dir + "/cov_inertia_mc.txt",arma::raw_ascii);
	results_inertia.save(output_dir + "/inertia_spread.txt",arma::raw_ascii);
	results_inertia_mean.save(output_dir + "/inertia_mean.txt",arma::raw_ascii);



	cov_moments_mc.save(output_dir + "/cov_moments_mc.txt",arma::raw_ascii);
	results_moments.save(output_dir + "/moments_spread.txt",arma::raw_ascii);
	results_moments_mean.save(output_dir + "/moments_mean.txt",arma::raw_ascii);

	

	cov_dims_mc.save(output_dir + "/cov_dims_mc.txt",arma::raw_ascii);
	results_dims.save(output_dir + "/dims_spread.txt",arma::raw_ascii);
	results_dims_mean.save(output_dir + "/dims_mean.txt",arma::raw_ascii);



	cov_mrp_mc.save(output_dir + "/cov_mrp_mc.txt",arma::raw_ascii);
	results_mrp.save(output_dir + "/mrp_spread.txt",arma::raw_ascii);
	results_mrp_mean.save(output_dir + "/mrp_mean.txt",arma::raw_ascii);




















	return 0;
}