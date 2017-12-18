#include "NavigationFilter.hpp"



NavigationFilter::NavigationFilter(const Args & args) : ExtendedKalmanFilter(args){
// 
}

int NavigationFilter::run(
	unsigned int N_iter,
	const arma::vec & X0_true,
	const arma::vec & X_bar_0,
	arma::vec & X0_true_small_body,
	arma::vec & X0_estimated_small_body,
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



	// Containers
	std::vector<arma::vec> X_small_body_true;
	std::vector<arma::vec> X_small_body_estimated;


	// The true state history of the spacecraft is computed
	this -> compute_true_state_history(X0_true,T_obs);

	// The true attitude of the small body is propagated
	this -> compute_true_small_body_attitude(X_small_body_true,X0_true_small_body,T_obs);

	// The estimated attitude of the small body is propagated
	this -> compute_estimated_small_body_attitude(X_small_body_estimated,
		X0_estimated_small_body,T_obs);



	// The filter is initialized
	if (this -> info_mat_bar_0.n_rows == 0){
		throw(std::runtime_error("The Navigation Filter needs an a-priori covariance matrix at initialization"));
	}

	arma::vec X_hat = X_bar_0;
	arma::mat P_hat = arma::inv(this -> info_mat_bar_0);

	for (unsigned int t = 0; t < T_obs.size() - 1; ++t){

		arma::mat dcm_LB = arma::zeros<arma::mat>(3,3);
		dcm_LB.col(0) = arma::normalise(this -> true_state_history[t].rows(0,2));
		dcm_LB.col(2) =  arma::normalise(arma::cross(this -> true_state_history[t].rows(0,2),
			this -> true_state_history[t].rows(3,5)));
		dcm_LB.col(1) = arma::normalise(arma::cross(dcm_LB.col(2),dcm_LB.col(0)));
		arma::inplace_trans(dcm_LB);

		(*args.get_mrp_BN_true()) = X_small_body_true[t].rows(0,2);
		(*args.get_mrp_BN_estimated()) = X_small_body_estimated[t].rows(0,2);
		(*args.get_mrp_LN_true()) = RBK::dcm_to_mrp(dcm_LB * RBK::mrp_to_dcm(*args.get_mrp_BN_true()));
		(*args.get_true_pos()) = this -> true_state_history[t].rows(0,2);

		// If true, there is an observation at the initial time
		if (t == 0 && T_obs[0] == 0){

			arma::vec Y_true_from_lidar = Observations::obs_pos_lidar(
				T_obs[0],this -> true_state_history[0],this -> args);

				// The prefit residual is computed
			auto y_bar = this -> compute_residual(0,X_hat,
				Y_true_from_lidar);

				// The measurement update is performed
			this -> measurement_update(0,X_hat, P_hat,
				y_bar,*this -> args.get_lidar_position_covariance_ptr());

				// The postfit residual is computed 
			auto y_hat = this -> compute_residual(0,X_hat,
				Y_true_from_lidar);

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


		dcm_LB.col(0) = arma::normalise(this -> true_state_history[t + 1].rows(0,2));
		dcm_LB.col(2) =  arma::normalise(arma::cross(this -> true_state_history[t + 1].rows(0,2),
			this -> true_state_history[t + 1].rows(3,5)));
		dcm_LB.col(1) = arma::normalise(arma::cross(dcm_LB.col(2),dcm_LB.col(0)));
		arma::inplace_trans(dcm_LB);


		(*args.get_mrp_BN_true()) = X_small_body_true[t + 1].rows(0,2);
		(*args.get_mrp_BN_estimated()) = X_small_body_estimated[t + 1].rows(0,2);
		(*args.get_mrp_LN_true()) = RBK::dcm_to_mrp(dcm_LB * RBK::mrp_to_dcm(*args.get_mrp_BN_true()));
		(*args.get_true_pos()) = this -> true_state_history[t + 1].rows(0,2);


		arma::vec Y_true_from_lidar = Observations::obs_pos_lidar(
			T_obs[t + 1],this -> true_state_history[t + 1],
			this -> args);

				// The prefit residual is computed
		auto y_bar = this -> compute_residual(T_obs[t+1],X_hat,
			Y_true_from_lidar);

				// The measurement update is performed
		this -> measurement_update(T_obs[t+1],X_hat, P_hat,
			y_bar,*this -> args.get_lidar_position_covariance_ptr());

				// The postfit residual is computed 
		auto y_hat = this -> compute_residual(T_obs[t+1],X_hat,
			Y_true_from_lidar);


			// S     ]] 
		this -> estimated_state_history.push_back(X_hat);
		this -> residuals.push_back(y_hat);
		this -> estimated_covariance_history.push_back(P_hat);

	}

	return 0;


}



void NavigationFilter::compute_true_small_body_attitude(std::vector<arma::vec> & X_small_body_true,
	arma::vec & X0_true_small_body,
	const std::vector<double> & T_obs){
	
	unsigned int N_true = X0_true_small_body.n_rows;

	// Set active inertia here
	this -> args.set_active_inertia(this -> args.get_true_shape_model() -> get_inertia());


	System dynamics(this -> args,
		N_true,
		Dynamics::attitude_dxdt );

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 , 1.0e-16 );

	auto tbegin = T_obs.begin();
	auto tend = T_obs.end();

	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_true_small_body, tbegin, tend,1e-10,
		Observer::push_back_attitude_state(X_small_body_true));


}


void NavigationFilter::compute_estimated_small_body_attitude(std::vector<arma::vec> & X_small_body_estimated,
	arma::vec & X0_estimated_small_body,
	const std::vector<double> & T_obs){
	unsigned int N_true = X0_estimated_small_body.n_rows;

	// Set active inertia here

	// Set active inertia here
	this -> args.set_active_inertia(this -> args.get_estimated_shape_model() -> get_inertia());


	System dynamics(this -> args,
		N_true,
		Dynamics::attitude_dxdt );

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 , 1.0e-16 );

	auto tbegin = T_obs.begin();
	auto tend = T_obs.end();

	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_estimated_small_body, tbegin, tend,1e-10,
		Observer::push_back_attitude_state(X_small_body_estimated));


}

