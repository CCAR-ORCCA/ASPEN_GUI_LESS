#include "BatchFilter.hpp"
#include "DebugFlags.hpp"
#include "System.hpp"
#include "Observer.hpp"
#include <boost/numeric/odeint.hpp>
#include "FixVectorSize.hpp"

BatchFilter::BatchFilter(const Args & args) : Filter(args){
}

int  BatchFilter::run(
	unsigned int N_iter,
	const arma::vec & X0_true,
	const arma::vec & X_bar_0,
	const std::vector<double> & T_obs,
	const arma::mat & R,
	const arma::mat & Q) {

	
	#if BATCH_DEBUG || FILTER_DEBUG
	std::cout << "- Running filter" << std::endl;
	#endif

	#if BATCH_DEBUG || FILTER_DEBUG
	
	std::cout << "-- Computing true observations" << std::endl;
	#endif


	this -> true_state_history.push_back(X0_true);

	// The true, noisy observations are computed
	this -> compute_true_observations(T_obs,args.get_sd_noise(),args.get_sd_noise_prop());

	auto true_ranges = this -> true_obs_history[0];


	#if BATCH_DEBUG || FILTER_DEBUG
	std::cout << "-- Done computing true observations" << std::endl;
	#endif


	// Containers
	arma::vec X_bar;
	arma::vec y_bar;

	arma::mat H;
	arma::mat H_Pcc_H;


	std::vector<double> sigma_consider_vector_ptr;
	std::vector<double> biases_consider_vector_ptr;

	this -> args.set_sigma_consider_vector_ptr(&sigma_consider_vector_ptr);
	this -> args.set_biases_consider_vector_ptr(&biases_consider_vector_ptr);


	arma::mat info_mat;
	arma::vec normal_mat;
	arma::vec dx_bar_0 = arma::zeros<arma::vec>(this -> true_state_history[0].n_rows);
	arma::vec dx_hat_consider = arma::zeros<arma::vec>(this -> true_state_history[0].n_rows);
	arma::mat P_hat_0;

	// The filter is initialized
	X_bar = X_bar_0;

	this -> info_mat_bar_0 = arma::zeros<arma::mat>(this -> true_state_history[0].n_rows,this -> true_state_history[0].n_rows);
	
	int iterations = N_iter;

	#if BATCH_DEBUG || FILTER_DEBUG
	std::cout << "-- Iterating the filter" << std::endl;
	#endif

	double old_residuals = std::numeric_limits<double>::infinity();

	// The batch is iterated
	for (unsigned int i = 0; i <= N_iter; ++i){


		sigma_consider_vector_ptr.clear();
		biases_consider_vector_ptr.clear();


		#if BATCH_DEBUG || FILTER_DEBUG
		std::cout << "--- Iteration " << i + 1 << "/" << N_iter << std::endl;
		#endif
		
		#if BATCH_DEBUG || FILTER_DEBUG
		std::cout << "----  Computing prefit residuals" << std::endl;
		#endif

		// The prefit residuals are computed
		this -> compute_prefit_residuals(T_obs[0],X_bar,y_bar);

		#if BATCH_DEBUG || FILTER_DEBUG
		std::cout << "----  Done computing prefit residuals" << std::endl;
		#endif

		// If the batch was only run for the pass-trough
		if (N_iter == 0){

			try{
				P_hat_0 = arma::inv(this -> info_mat_bar_0);
			}

			catch (std::runtime_error & e){
				P_hat_0.set_size(arma::size(this -> info_mat_bar_0));
				P_hat_0.fill(arma::datum::nan);
			}

			break;
		}

		double N_mes, rms_res;
		
			#if BATCH_DEBUG || FILTER_DEBUG

		arma::vec y_non_zero = y_bar.elem(arma::find(y_bar));

		rms_res = std::sqrt(std::pow(arma::norm(y_non_zero),2) / y_non_zero.n_rows) /T_obs.size();
		N_mes = y_non_zero.n_rows;

		std::cout << "-----  Residuals: " << rms_res << std::endl;
			#endif
		

		// The normal and information matrices are assembled
		info_mat = this -> info_mat_bar_0;
		normal_mat = this ->  info_mat_bar_0 * dx_bar_0;


		#if BATCH_DEBUG || FILTER_DEBUG
		std::cout << "----  Assembling normal equations" << std::endl;
		#endif

		// H has already been pre-multiplied by the corresponding gains
		H = this -> estimate_jacobian_observations_fun(T_obs[0], X_bar ,this -> args);

		arma::sp_mat P_cc(H.n_rows,H.n_rows);
		arma::sp_mat W(H.n_rows,H.n_rows);

		arma::vec biases = arma::zeros<arma::vec>(H.n_rows);


		#if BATCH_DEBUG || FILTER_DEBUG
		std::cout << "----  Populating consider covariance and removing outliers" << std::endl;
		#endif

		for (unsigned int p = 0; p < H.n_rows; ++p){

			P_cc(p,p) = std::pow(sigma_consider_vector_ptr[p],2);

			double mes_range;
			
			if (true_ranges.subvec(p,p).has_nan()){
				mes_range = 0;
			}
			else{
				mes_range = true_ranges(p);
			}
			

			W(p,p) = 1./(std::pow(args.get_sd_noise() + args.get_sd_noise_prop() * mes_range,2));
			std::cout << W(p,p) << std::endl;
			biases(p) = biases_consider_vector_ptr[p];

			if (std::abs(y_bar(p)) > 10 * rms_res){
				H.row(p).fill(0);
				y_bar(p) = 0;
			}
		}
		#if BATCH_DEBUG || FILTER_DEBUG
		std::cout << "---- Larger residuals after removing outliers: " << arma::abs(y_bar).max() << std::endl;
		#endif


		// The weighting matrix is computed


		info_mat += H.t() * W * H ;
		normal_mat += H.t() * W * y_bar ;



		H_Pcc_H = H.t() * P_cc * H;

		// The deviation is solved
		auto dx_hat = arma::solve(info_mat,normal_mat);

		// The covariance of the state at the initial time is computed
		P_hat_0 = arma::inv(info_mat) ;

		// dx_hat_consider = - P_hat_0 * H.t() / std::pow(args.get_sd_noise(),2) * biases;

		#if BATCH_DEBUG || FILTER_DEBUG
		std::cout << "--- Info mat: \n" << info_mat << std::endl;
		std::cout << "--- Normal mat:\n " << normal_mat << std::endl;
		std::cout << "---  Deviation: "<< std::endl;
		std::cout << dx_hat << std::endl;
		std::cout << "---  Deviation with consider effect: "<< std::endl;
		std::cout << dx_hat + dx_hat_consider<< std::endl;
		#endif

		
		// The deviation is applied to the state
		arma::vec X_hat_0 = X_bar + dx_hat + dx_hat_consider;

		X_bar = X_hat_0;


		// The a-priori deviation is adjusted
		dx_bar_0 = dx_bar_0 - dx_hat - dx_hat_consider;


		// Checking for convergence
		double variation = std::abs(rms_res - old_residuals)/rms_res * 100;
		if (variation < 1e-3){

		#if BATCH_DEBUG || FILTER_DEBUG
			std::cout << "--- Batch Filter has converged" << std::endl;
		#endif

			iterations = i;
			break;
		}

		else{
		#if BATCH_DEBUG || FILTER_DEBUG

			std::cout << "--- Variation in residuals: " << variation << " %" << std::endl;
		#endif

			old_residuals = rms_res;
		}


	}


	std::cout << "-- State Covariance \n";
	std::cout << P_hat_0 << std::endl;

	// This is where the covariance should be augmented with its 
	// consider component

	// P_hat_0 += 1./std::pow(args.get_sd_noise(),4) * P_hat_0 * H_Pcc_H * P_hat_0;


	std::cout << "-- Consider State Covariance \n";
	std::cout << P_hat_0 << std::endl;

	// The results are saved

	this -> estimated_state_history.push_back(X_bar);
	this -> estimated_covariance_history.push_back(P_hat_0);

	this -> residuals.push_back( y_bar);

	y_bar.save("../output/range_residuals/residuals_" + std::to_string(T_obs[0])+ ".txt",arma::raw_ascii);

	#if BATCH_DEBUG || FILTER_DEBUG

	std::cout << "-- Exiting batch "<< std::endl;
	#endif

	return iterations;

}

void BatchFilter::compute_prefit_residuals(
	double t,
	const arma::vec & X_bar,
	arma::vec & y_bar){

	// The previous residuals are discarded
	double rms_res = 0;


	arma::vec true_obs = this -> true_obs_history[0];
	arma::vec computed_obs = this -> estimate_observation_fun(t, X_bar ,this -> args);

		// Potentials nan are looked for and turned into zeros
		// the corresponding row in H will be set to zero
	arma::vec residual = true_obs - computed_obs;

	#pragma omp parallel for
	for (unsigned int i = 0; i < residual.n_rows; ++i){
		if (residual.subvec(i,i).has_nan() || std::abs(residual(i)) > 1e10 ){
			residual(i) = 0;

		}
	}

	y_bar = residual;

	arma::vec y_non_zero = residual.elem(arma::find(residual));
		#if BATCH_DEBUG
	std::cout << " - Usable residuals in batch: " << y_non_zero.n_rows << std::endl;
	std::cout << " - Largest residual in batch: " << arma::abs(y_bar).max() << std::endl;
		#endif
	rms_res += std::sqrt(std::pow(arma::norm(y_non_zero),2) / y_non_zero.n_rows);


}





