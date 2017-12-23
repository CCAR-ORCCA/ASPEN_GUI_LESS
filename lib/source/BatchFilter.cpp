#include "BatchFilter.hpp"
BatchFilter::BatchFilter(const Args & args) : Filter(args){
}

int  BatchFilter::run(
	unsigned int N_iter,
	const arma::vec & X0_true,
	const arma::vec & X_bar_0,
	const std::vector<double> & T_obs,
	const arma::mat & R,
	const arma::mat & Q,
	bool verbose) {

	if (T_obs.size() == 0){
		return -1;
	}

	if (verbose){
		std::cout << "- Running filter" << std::endl;
		std::cout << "-- Computing true state history" << std::endl;
	}

	// The true state history is computed
	this -> compute_true_state_history(X0_true,T_obs);
	if (verbose){
		std::cout << "-- Computing true observations" << std::endl;
	}

	// The true, noisy observations are computed
	this -> compute_true_observations(T_obs,R);

	if (verbose){
		std::cout << "-- Done computing true observations" << std::endl;
	}


	// Containers
	std::vector<arma::vec> X_bar;
	std::vector<arma::mat> stm;
	std::vector<arma::vec> Y_bar;
	std::vector<arma::vec> y_bar;

	arma::mat H;

	arma::mat info_mat;
	arma::vec normal_mat;
	arma::vec dx_bar_0 = arma::zeros<arma::vec>(this -> true_state_history[0].n_rows);
	arma::mat P_hat_0;

	bool has_converged;

	// The filter is initialized
	X_bar.push_back(X_bar_0);
	arma::mat W = arma::inv(R);

	if (this -> info_mat_bar_0.n_cols == 0 && this -> info_mat_bar_0.n_rows == 0){
		this -> info_mat_bar_0 = arma::zeros<arma::mat>(X_bar_0.n_rows,X_bar_0.n_rows);
	}

	int iterations = N_iter;

	if (verbose){
		std::cout << "-- Iterating the filter" << std::endl;
	}


	// The batch is iterated
	for (unsigned int i = 0; i <= N_iter; ++i){

		if (verbose){
			std::cout << "--- Iteration " << i + 1 << "/" << N_iter << std::endl;
		}

		
		if (verbose){
			std::cout << "----  Computing reference trajectory" << std::endl;
		}

		// The reference trajectory is computed along with the STM
		this -> compute_reference_state_history(T_obs,X_bar,stm);

		if (verbose){
			std::cout << "----  Computing prefit residuals" << std::endl;
		}

		// The prefit residuals are computed
		this -> compute_prefit_residuals(T_obs,
			X_bar,
			y_bar,
			has_converged
			,R);

		if (verbose){
			std::cout << "----  Done computing prefit residuals" << std::endl;
			std::cout << "----  Has converged? " << has_converged << std::endl;
		}

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

		// The state is checked for convergence based on the residuals
		if (has_converged && i != 0){
			iterations = i;
			break;
		}
		else{
			if (verbose){

				arma::vec y_non_zero = y_bar.back().elem(arma::find(y_bar.back()));

				double rms_res = std::sqrt(std::pow(arma::norm(y_non_zero),2) / y_non_zero.n_rows) /T_obs.size();



				std::cout << "-----  Has not converged" << std::endl;
				std::cout << "-----  Residuals: " << rms_res << std::endl;
			}
		}

		// The normal and information matrices are assembled
		info_mat = this -> info_mat_bar_0;
		normal_mat = this ->  info_mat_bar_0 * dx_bar_0;


		if (verbose){
			std::cout << "----  Assembling normal equations" << std::endl;
		}

		for (unsigned int p = 0; p < T_obs.size(); ++ p){

			if (arma::norm(y_bar[p]) == 0){
				// A residual can't have a zero norm. This means that no observation was available
				continue;
			}


			H = this -> estimate_jacobian_observations_fun(T_obs[p], X_bar[p] ,
				this -> args) * stm[p];

			if (W.n_rows > 1){
				info_mat += H.t() * W * H;
				normal_mat += H.t() * W * y_bar[p];
			}
			else{
				info_mat += H.t() * W(0,0) * H;
				normal_mat += H.t() * W(0,0) * y_bar[p];
			}
		}


		// The deviation is solved
		auto dx_hat = arma::solve(info_mat,normal_mat);

		if (verbose){
			std::cout << "---  Deviation: "<< std::endl;
			std::cout << dx_hat << std::endl;
		}

		// The covariance of the state at the initial time is computed
		P_hat_0 = arma::inv(info_mat);

		// The deviation is applied to the state

		arma::vec X_hat_0 = X_bar[0] + dx_hat;

		X_bar[0] = X_hat_0;


		// The a-priori deviation is adjusted
		dx_bar_0 = dx_bar_0 - dx_hat;

	}

	// The results are saved
	this -> compute_reference_state_history(T_obs,X_bar,stm);
	this -> compute_covariances(P_hat_0,stm);

	this -> estimated_state_history = X_bar;
	this -> stm_history = stm;
	
	this -> residuals = y_bar;

	if (verbose){
			std::cout << "-- Exiting batch "<< std::endl;
		}

	return iterations;

}

void BatchFilter::compute_prefit_residuals(
	const std::vector<double> & T_obs,
	const std::vector<arma::vec> & X_bar,
	std::vector<arma::vec> & y_bar,
	bool & has_converged,
	const arma::mat & R){

	// The previous residuals are discarded
	y_bar.clear();
	double rms_res = 0;

	// The new residuals are computed
	for (unsigned int p = 0; p < T_obs.size(); ++ p){

		arma::vec true_obs = this -> true_obs_history[p];
		arma::vec computed_obs = this -> estimate_observation_fun(T_obs[p], X_bar[p] ,this -> args);

		// Potentials nan are looked for and turned into zeros
		// the corresponding row in H will be set to zero
		arma::vec residual = true_obs - computed_obs;

		#pragma omp parallel for
		for (unsigned int i = 0; i < residual.n_rows; ++i){
			if (residual.subvec(i,i).has_nan() || std::abs(residual(i)) > 1e10 ){
				residual(i) = 0;

			}
		}

		y_bar.push_back(residual);

		arma::vec y_non_zero = residual.elem(arma::find(residual));
		
		rms_res += std::sqrt(std::pow(arma::norm(y_non_zero),2) / y_non_zero.n_rows) /T_obs.size();

	}

	// The convergence is checked. 
	if (rms_res < 1.1 * std::sqrt(R(0,0))){
		has_converged = true;

	}
	else{
		has_converged = false;
	}

}

void BatchFilter::compute_reference_state_history(
	const std::vector<double> & T_obs,
	std::vector<arma::vec> & X_bar,
	std::vector<arma::mat> & stm){

	// If the estimated state has no dynamics,
	// all the STMs are equal to the identity and the state is constant
	if (this -> estimate_dynamics_fun == nullptr){

		arma::vec X_bar_0 = X_bar[0];

		X_bar.clear();
		stm.clear();

		X_bar.push_back(X_bar_0);

		// X_bar already has one state in - the a-priori
		stm.push_back(arma::eye<arma::mat>(X_bar[0].n_rows,X_bar[0].n_rows));

		for (unsigned int i = 1; i < T_obs.size(); ++i){
			X_bar.push_back(X_bar[0]);
			stm.push_back(stm[0]);
		}
		

	}

	// The state/STM are propagated
	else{


		//
		// Odeint is called here. the state is propagated
		// along with the state transition matrices
		//

		unsigned int N_est = X_bar[0].n_rows;
		unsigned int N_true = 0;

		arma::vec X_bar_0 = X_bar[0];

		System dynamics(this -> args,
			N_est,
			this -> estimate_dynamics_fun ,
			this -> estimate_jacobian_dynamics_fun,
			N_true,
			nullptr );
		

		arma::vec x(N_est + N_est * N_est);
		x.rows(0,N_est - 1) = X_bar[0];
		x.rows(N_est,N_est + N_est * N_est - 1) = arma::vectorise(arma::eye<arma::mat>(N_est,N_est));

		std::vector<arma::vec> augmented_state_history;
		typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
		auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,
			1.0e-16 );
		
		auto tbegin = T_obs.begin();
		auto tend = T_obs.end();
		boost::numeric::odeint::integrate_times(stepper, dynamics, x, tbegin, tend,1e-10,
			Observer::push_back_state(augmented_state_history));

		
		X_bar.clear();
		stm.clear();

		for (unsigned int i = 0; i < T_obs.size(); ++i){
			X_bar.push_back(augmented_state_history[i].rows(0,N_est - 1));
			stm.push_back(arma::reshape(augmented_state_history[i].rows(N_est,N_est + N_est * N_est - 1),N_est,N_est));
		}
		


	}

}

void BatchFilter::compute_covariances(const arma::mat & P_hat_0,
	const std::vector<arma::mat> & stm){

	this -> estimated_covariance_history.clear();
	for (unsigned int i = 0; i < stm.size(); ++i){
		auto phi = stm.at(i);
		this -> estimated_covariance_history.push_back(phi * P_hat_0 * phi.t());
	}

}



void BatchFilter::set_gamma_fun(arma::mat (*gamma_fun)(double dt)){
// Does nothing
}





