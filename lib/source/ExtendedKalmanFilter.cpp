#include "ExtendedKalmanFilter.hpp"



ExtendedKalmanFilter::ExtendedKalmanFilter(const Args & args) : SequentialFilter(args){
// 
}

int ExtendedKalmanFilter::run(
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

	// The filter is initialized

	if (this -> info_mat_bar_0.n_rows == 0){
		throw(std::runtime_error("The EKF needs an a-priori covariance matrix at initialization"));
	}

	arma::vec X_hat = X_bar_0;
	arma::mat P_hat = arma::inv(this -> info_mat_bar_0);

	arma::vec previous_norm_res_squared = arma::zeros<arma::vec>(this -> true_obs_history[0].n_rows);

	int iterations = N_iter;

	if (verbose){
		std::cout << "-- Iterating the filter" << std::endl;
	}

	// The EKF is iterated
	for (unsigned int iter = 0; iter < N_iter; ++iter){

		for (unsigned int t = 0; t < T_obs.size() - 1; ++t){

			// If true, there is an observation at the initial time
			if (t == 0 && T_obs[0] == 0){

				// The prefit residual is computed
				auto y_bar = this -> compute_residual(0,X_hat,
					this -> true_obs_history[0]);

				// The measurement update is performed
				this -> measurement_update(0,X_hat, P_hat,
					y_bar,R);

				// The postfit residual is computed 
				auto y_hat = this -> compute_residual(0,X_hat,
					this -> true_obs_history[0]);


				// STORE RESULTS 
				this -> estimated_state_history.push_back(X_hat);
				this -> residuals.push_back(y_hat);
				this -> estimated_covariance_history.push_back(P_hat);
			}
			else if (t == 0){
				// STORE RESULTS 
				this -> estimated_state_history.push_back(X_hat);
				this -> estimated_covariance_history.push_back(P_hat);
			}

			// The a-priori is propagated until the next timestep
			this -> time_update(T_obs[t],T_obs[t + 1],X_hat,P_hat);

			// SNC is applied 
			this -> apply_SNC(T_obs[t + 1] - T_obs[t],P_hat,Q);

			// The prefit residual is computed
			auto y_bar = this -> compute_residual(T_obs[t + 1],X_hat,
				this -> true_obs_history[t + 1]);
			
			// The measurement update is performed
			this -> measurement_update(T_obs[t + 1],X_hat, P_hat,
				y_bar,R);

			// The postfit residual is computed 
			auto y_hat = this -> compute_residual(T_obs[t + 1],X_hat,
				this -> true_obs_history[t + 1]);

			// STORE RESULTS 
			this -> estimated_state_history.push_back(X_hat);
			this -> residuals.push_back(y_hat);
			this -> estimated_covariance_history.push_back(P_hat);

		}

		


		// The convergence of the EKF is checked
		bool has_converged = false;
		
		if (has_converged){
			
			if (verbose){
				std::cout << "Converged in " << iter + 1 << " iterations\n";
			}
			break;

		}

		else if(iter == N_iter - 1){
			if (verbose){
				std::cout << "Stalled at " << iter + 1 << " iterations\n";
			}
			break;

		}

		else{

			throw(std::runtime_error("EKF iteration not yet implemented"));

		}

	}

	return iterations;


}




void ExtendedKalmanFilter::time_update(double t_now, double t_next,
	arma::vec & X_hat, arma::mat & P_hat) const{

	unsigned int N_est = X_hat.n_rows;
	unsigned int N_true = 0;

	System dynamics(this -> args,
		N_est,
		this -> estimate_dynamics_fun ,
		this -> estimate_jacobian_dynamics_fun,
		N_true,
		nullptr );

	arma::vec x(N_est + N_est * N_est);
	x.rows(0,N_est - 1) = X_hat;
	x.rows(N_est,N_est + N_est * N_est - 1) = arma::vectorise(arma::eye<arma::mat>(N_est,N_est));

	std::vector<arma::vec> augmented_state_history;
	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,
		1.0e-16 );

	std::vector<double> times;
	times.push_back(t_now);
	times.push_back(t_next);

	auto tbegin = times.begin();
	auto tend = times.end();


	boost::numeric::odeint::integrate_times(stepper, dynamics, x, tbegin, tend,1e-10,
		Observer::push_back_augmented_state(augmented_state_history));

	if (augmented_state_history.size() != 2){
		throw(std::runtime_error("augmented_state_history should have two elements only"));
	}

	for (unsigned int i = 0; i < N_est; ++i){
		X_hat(i) = augmented_state_history[1](i);
	}
	

	arma::mat stm = arma::reshape(
		augmented_state_history[1].rows(N_est,N_est + N_est * N_est - 1),
		N_est,N_est);

	

	P_hat = stm * P_hat * stm.t();


}

void ExtendedKalmanFilter::measurement_update(double t,arma::vec & X_bar, arma::mat & P_bar,
	const arma::vec & res,const arma::mat & R) const{

	auto H = this -> estimate_jacobian_observations_fun(t, X_bar , this -> args);

	// The Kalman gain is computed
	arma::mat K = P_bar * H.t() * arma::inv(H * P_bar * H.t() + R);

	// The innovation is added to the state
	X_bar = X_bar + K * res;
	

	// The covariance is updated
	auto I = arma::eye<arma::mat>(X_bar.n_rows,X_bar.n_rows);
	P_bar = (I - K * H) * P_bar * (I - K * H).t() + K * R * K.t();

}