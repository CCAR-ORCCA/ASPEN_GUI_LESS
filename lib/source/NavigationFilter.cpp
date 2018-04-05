#include "NavigationFilter.hpp"
#include "DebugFlags.hpp"
#include "System.hpp"
#include "Observer.hpp"
#include "Dynamics.hpp"
#include <boost/numeric/odeint.hpp>

NavigationFilter::NavigationFilter(const Args & args) : ExtendedKalmanFilter(args){
// 
}

int NavigationFilter::run(
	unsigned int N_iter,
	const arma::vec & X0_true_augmented,
	const arma::vec & X0_estimated_augmented,
	const std::vector<double> & T_obs,
	const arma::mat & R,
	const arma::mat & Q) {

	if (T_obs.size() == 0){
		return -1;
	}

	#if NAVIGATION_DEBUG
	std::cout << "-- Computing true navigation state history" << std::endl;
	#endif

	this -> compute_true_state(T_obs,X0_true_augmented,true);

	#if NAVIGATION_DEBUG
	std::cout << "-- Computing estimated small body state history" << std::endl;
	#endif

	// The filter is initialized
	if (this -> info_mat_bar_0.n_rows == 0){
		throw(std::runtime_error("The Navigation Filter needs an a-priori covariance matrix at initialization"));
	}

	// Containers
	arma::vec X_hat = X0_estimated_augmented;
	arma::mat P_hat = arma::inv(this -> info_mat_bar_0);
	arma::mat dcm_LB = arma::zeros<arma::mat>(3,3);

	#if NAVIGATION_DEBUG
	std::cout << "-- Running the Navigation Filter" << std::endl;

	#endif
	for (unsigned int t = 0; t < T_obs.size() - 1; ++t){

		#if NAVIGATION_DEBUG
		std::cout << "##################### Time : " << T_obs[t] << " Index: " << t << "/" << T_obs.size() - 2 <<std::endl;
		#endif


		// If true, there is an observation at the initial time
		if (t == 0){

			assert(this -> estimated_state_history.is_empty());
			assert(this -> estimated_covariance_history.is_empty());

			this -> set_states(X_hat,this -> true_state_history[t],t);

			// 
			arma::vec Y_true_from_lidar = this -> true_observation_fun(T_obs[0],X_hat,this -> args);

				// The prefit residual are computed
			auto y_bar = this -> compute_residual(0,X_hat,Y_true_from_lidar);

				// The measurement update is performed
			this -> measurement_update(0,X_hat, P_hat,y_bar,*this -> args.get_lidar_position_covariance_ptr());
			this -> set_states(X_hat,this -> true_state_history[t],t);

				// The postfit residual are computed 
			auto y_hat = this -> compute_residual(0,X_hat,Y_true_from_lidar);

				// STORE RESULTS 
			this -> estimated_state_history.push_back(X_hat);
			this -> residuals.push_back(y_hat);
			this -> estimated_covariance_history.push_back(P_hat);
		}
		
		std::cout << "State error: " << std::endl;
		std::cout << (this -> true_state_history[t] - X_hat).t() << std::endl;

		// The a-priori is propagated until the next timestep
		this -> time_update(T_obs[t],T_obs[t + 1],X_hat,P_hat);

	

		// SNC is applied 
		this -> apply_SNC(T_obs[t + 1] - T_obs[t],P_hat,Q);
		this -> set_states(X_hat,this -> true_state_history[t+1],t + 1);

		arma::vec Y_true_from_lidar = this -> true_observation_fun(T_obs[t + 1],X_hat,this -> args);

		// The prefit residual is computed
		auto y_bar = this -> compute_residual(T_obs[t+1],X_hat,Y_true_from_lidar);

		// The measurement update is performed
		this -> measurement_update(T_obs[t+1],X_hat, P_hat,y_bar,
			*this -> args.get_lidar_position_covariance_ptr());
		this -> set_states(X_hat,this -> true_state_history[t],t);

		// The postfit residual is computed 
		auto y_hat = this -> compute_residual(T_obs[t+1],X_hat,Y_true_from_lidar);

			// 
		this -> estimated_state_history.push_back(X_hat);
		this -> residuals.push_back(y_hat);
		this -> estimated_covariance_history.push_back(P_hat);

	}

	#if NAVIGATION_DEBUG
	std::cout << "-- Leaving" << std::endl;
	#endif

	return 0;


}


void NavigationFilter::set_states(const arma::vec & X_hat,arma::vec X_true,unsigned int t){
	

	this -> args.set_estimated_pos(X_hat.subvec(0,2));
	this -> args.set_estimated_vel(X_hat.subvec(3,5));
	this -> args.set_estimated_mrp_BN(X_hat.subvec(6,8));

	this -> args.set_true_pos(X_true.subvec(0,2));
	this -> args.set_true_vel(X_true.subvec(3,5));
	this -> args.set_true_mrp_BN(X_true.subvec(6,8));


	// std::cout << this -> args.get_estimated_pos().t() << std::endl;
	// std::cout << this -> args.get_estimated_vel().t() << std::endl;
	// std::cout << this -> args.get_estimated_mrp_BN().t() << std::endl;
	// std::cout << this -> args.get_true_pos().t() << std::endl;
	// std::cout << this -> args.get_true_vel().t() << std::endl;
	// std::cout << this -> args.get_true_mrp_BN().t() << std::endl << std::endl;




}


void NavigationFilter::compute_true_state(std::vector<double> T_obs,
	const arma::vec & X0_true_augmented,
	bool save){

	// Containers
	std::vector<arma::vec> X_augmented;
	auto N_true = X0_true_augmented.n_rows;
	arma::vec X0_true_augmented_cp(X0_true_augmented);

	this -> true_state_history.clear();


	// Set active inertia here
	this -> args.set_active_inertia(this -> args.get_true_shape_model() -> get_inertia());

	System dynamics(this -> args,N_true,this -> true_dynamics_fun );

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

	auto tbegin = T_obs.begin();
	auto tend = T_obs.end();

	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_true_augmented_cp, tbegin, tend,1e-3,
		Observer::push_back_augmented_state(this -> true_state_history));

	

	if (save == true){

		arma::mat true_spacecraft_state_mat = arma::mat(6,this -> true_state_history.size());
		arma::mat true_asteroid_state_mat = arma::mat(6,this -> true_state_history.size());

		for (unsigned int i = 0; i < this -> true_state_history.size(); ++i){


			true_spacecraft_state_mat.col(i) = this -> true_state_history[i].rows(0,5);
			true_asteroid_state_mat.col(i) = this -> true_state_history[i].rows(6,11);

		}

		true_spacecraft_state_mat.save("true_spacecraft_state.txt",arma::raw_ascii);
		true_asteroid_state_mat.save("true_asteroid_state_mat.txt",arma::raw_ascii);

	}

}




