#include "BatchFilter.hpp"

BatchFilter::BatchFilter(const Args & args) : Filter(args){
}

int BatchFilter::run(
	unsigned int N_iter,
	const arma::vec & X0_true,
	const arma::vec & X_bar_0,
	const std::vector<double> & T_obs,
	const arma::mat & R) {

	if (T_obs.size() == 0){
		return -1;
	}

	// The true state history is computed
	this -> compute_true_state_history(X0_true,T_obs);

	// The true, noisy observations are computed
	this -> compute_true_observations(T_obs,R);

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
	arma::vec previous_norm_res_squared = arma::zeros<arma::vec>(this -> true_obs_history[0].n_rows);
	arma::mat W = arma::inv(R);

	if (this -> info_mat_bar_0.n_cols == 0 && this -> info_mat_bar_0.n_rows == 0){
		this -> info_mat_bar_0 = arma::zeros<arma::mat>(X_bar_0.n_rows,X_bar_0.n_rows);
	}

	int iterations = N_iter;
	
	// The batch is iterated
	for (unsigned int i = 0; i <= N_iter; ++i){

		// The reference trajectory is computed along with the STM
		this -> compute_reference_state_history(T_obs,X_bar,stm);

		// The prefit residuals are computed
		this -> compute_prefit_residuals(T_obs,
			X_bar,
			y_bar,
			has_converged,
			previous_norm_res_squared);

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
		if (has_converged){
			iterations = i;
			break;
		}

		// The normal and information matrices are assembled
		info_mat = this -> info_mat_bar_0;
		normal_mat = this ->  info_mat_bar_0 * dx_bar_0;


		for (unsigned int p = 0; p < T_obs.size(); ++ p){

			if (arma::norm(y_bar[p]) == 0){
				// A residual can't have a zero norm. This means that no observation was available
				continue;
			}

			H = this -> jacobian_observations_fun(T_obs[p], X_bar[p] ,this -> args) * stm[p];

			info_mat += H.t() * W * H;

			normal_mat += H.t() * W * y_bar[p];
		}

		// The deviation is solved
		auto dx_hat = arma::solve(info_mat,normal_mat);

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

	return iterations;

}

void BatchFilter::compute_prefit_residuals(
	const std::vector<double> & T_obs,
	const std::vector<arma::vec> & X_bar,
	std::vector<arma::vec> & y_bar,
	bool & has_converged,
	arma::vec & previous_norm_res_squared){

	// The previous residuals are discarded
	y_bar.clear();
	arma::vec norm_res_squared = arma::zeros<arma::vec>(this -> true_obs_history[0].n_rows);

	// The new residuals are computed
	for (unsigned int p = 0; p < T_obs.size(); ++ p){

		arma::vec residual = this -> true_obs_history[p] - this -> observation_fun(T_obs[p], X_bar[p] ,this -> args);
		if (residual.has_nan() == false){
			y_bar.push_back(residual);
		}
		else{
			y_bar.push_back(arma::zeros<arma::vec>(residual.n_cols));
		}

		norm_res_squared += arma::dot(y_bar.back(),y_bar.back())/T_obs.size();
	}

	// The convergence is checked. Could replace by norm?
	arma::vec rel_change = arma::abs(norm_res_squared - previous_norm_res_squared) / arma::norm(norm_res_squared);
	
	if (arma::max(rel_change) < 1e-8){
		has_converged = true;
	}
	
	else{
		has_converged = false;
		previous_norm_res_squared = norm_res_squared;
	}

}

void BatchFilter::compute_reference_state_history(
	const std::vector<double> & T_obs,
	std::vector<arma::vec> & X_bar,
	std::vector<arma::mat> & stm){

	// If the estimated state has no dynamics,
	// all the STMs are equal to the identity and the state is constant
	if (this -> estimate_dynamics_fun == nullptr){

		if (X_bar.size() == T_obs.size()){

			// The first state is copied accross 
			for (unsigned int i = 1; i < T_obs.size(); ++i){

				X_bar[i] = X_bar[0];
				stm[i] = stm[0];
			}	
		}
		else{
			// X_bar already has one state in - the a-priori
			stm.push_back(arma::eye<arma::mat>(X_bar[0].n_rows,X_bar[0].n_rows));

			for (unsigned int i = 1; i < T_obs.size(); ++i){
				X_bar.push_back(X_bar[0]);
				stm.push_back(stm[0]);
			}
		}

	}

	// The state/STM are propagated
	else{


		//
		// Odeint is called here. the state is propagated
		// along with the state transition matrices
		//

		std::vector<arma::vec> X_bar_from_integrator;
		std::vector<arma::mat> stm_from_integrator;


		if (X_bar.size() == T_obs.size()){
			for (unsigned int i = 0; i < T_obs.size(); ++i){
				X_bar[i]= (X_bar_from_integrator[i]);
				stm[i]= (stm_from_integrator[i]);
			}
		}

		else{
			for (unsigned int i = 0; i < T_obs.size(); ++i){
				X_bar.push_back(X_bar_from_integrator[i]);
				stm.push_back(stm_from_integrator[i]);
			}
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