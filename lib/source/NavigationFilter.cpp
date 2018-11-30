#include "NavigationFilter.hpp"
#include "SystemDynamics.hpp"
#include "Observer.hpp"
#include "Dynamics.hpp"
#include <boost/numeric/odeint.hpp>
#include <vector>
#define NAVIGATION_DEBUG 1

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

	// The filter is initialized
	if (this -> info_mat_bar_0.n_rows == 0){
		throw(std::runtime_error("The Navigation Filter needs an a-priori covariance matrix at initialization"));
	}

	// Containers
	arma::vec X_hat = X0_estimated_augmented;
	arma::mat P_hat = arma::inv(this -> info_mat_bar_0);
	arma::mat dcm_LB = arma::zeros<arma::mat>(3,3);
	arma::vec Y_true_from_lidar;

	#if NAVIGATION_DEBUG
	std::cout << "-- Running the Navigation Filter" << std::endl;

	#endif
	for (unsigned int t = 0; t < T_obs.size() - 1; ++t){

		#if NAVIGATION_DEBUG
		std::cout << "##################### Time : " << T_obs[t] << " Index: " << t << "/" << T_obs.size() - 2 <<std::endl;
		#endif


		// If true, there is an observation at the initial time
		if (t == 0){

			assert(this -> estimated_state_history.empty());
			assert(this -> estimated_covariance_history.empty());
			
			this -> iterated_measurement_update(t,T_obs,X_hat, P_hat);
		}
		
		std::cout << "State error before time update: " << std::endl;
		std::cout << (this -> true_state_history[t] - X_hat).t() << std::endl;

		// The a-priori is propagated until the next timestep
		this -> time_update(T_obs[t],T_obs[t + 1],X_hat,P_hat);

		std::cout << "State error after time update: " << std::endl;
		std::cout << (this -> true_state_history[t+1] - X_hat).t() << std::endl;

		// SNC is applied 
		this -> apply_SNC(T_obs[t + 1] - T_obs[t],P_hat,Q);

		this -> iterated_measurement_update(t + 1,T_obs,X_hat, P_hat);


	}

	#if NAVIGATION_DEBUG
	std::cout << "-- Leaving" << std::endl;
	#endif

	return 0;


}


void NavigationFilter::set_states(const arma::vec & X_hat,arma::vec X_true,const arma::mat & P_hat,unsigned int t){
	

	this -> args.set_estimated_pos(X_hat.subvec(0,2));
	this -> args.set_estimated_vel(X_hat.subvec(3,5));
	this -> args.set_estimated_mrp_BN(X_hat.subvec(6,8));

	this -> args.set_true_pos(X_true.subvec(0,2));
	this -> args.set_true_vel(X_true.subvec(3,5));
	this -> args.set_true_mrp_BN(X_true.subvec(6,8));

	this -> args.set_state_covariance(P_hat);

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


	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

	auto tbegin = T_obs.begin();
	auto tend = T_obs.end();

	boost::numeric::odeint::integrate_times(stepper, this -> true_dynamics_system, X0_true_augmented_cp, tbegin, tend,1e-3,
		Observer::push_back_augmented_state(this -> true_state_history));


	if (save == true){

		arma::mat true_spacecraft_state_mat = arma::mat(6,this -> true_state_history.size());
		arma::mat true_asteroid_state_mat = arma::mat(6,this -> true_state_history.size());

		for (unsigned int i = 0; i < this -> true_state_history.size(); ++i){


			true_spacecraft_state_mat.col(i) = this -> true_state_history[i].subvec(0,5);
			true_asteroid_state_mat.col(i) = this -> true_state_history[i].subvec(6,11);

		}

		true_spacecraft_state_mat.save("true_spacecraft_state.txt",arma::raw_ascii);
		true_asteroid_state_mat.save("true_asteroid_state_mat.txt",arma::raw_ascii);

	}

}






void NavigationFilter::iterated_measurement_update(unsigned int t,
	const std::vector<double> & T_obs,
	arma::vec & X_hat, 
	arma::mat & P_hat){


	bool done_iterating = false;
	auto N_iter = this -> args.get_N_iter_mes_update();
	arma::vec Y_true_from_lidar;
	double previous_mahalanobis_distance = std::numeric_limits<double>::infinity();

	// The measurement updated is iterated
	for (int i =0; i < N_iter; ++i ){
		std::cout << "---- EKF Iteration: " << i + 1 << "/" << N_iter << std::endl;

		if (i == N_iter - 1){
			done_iterating = true;
		}
		this -> set_states(X_hat,this -> true_state_history[t],P_hat,t);

		// The batch is called to compute a position/attitude measurement
		Y_true_from_lidar = this -> true_observation_fun(T_obs[t],X_hat,this -> args);

		// The prefit residual are computed
		auto y_bar = this -> compute_residual(T_obs[t],X_hat,Y_true_from_lidar);

			// The measurement update is performed
		this -> measurement_update(T_obs[t],
			X_hat, 
			P_hat,
			y_bar,
			*this -> args.get_batch_output_covariance_ptr(),
			done_iterating,
			previous_mahalanobis_distance
			);
		this -> set_states(X_hat,this -> true_state_history[t],P_hat,t);
		
		if (done_iterating){
			break;
		}
	}


	// The postfit residual are computed 
	arma::vec y_hat = this -> compute_residual(T_obs[t],X_hat,Y_true_from_lidar);

	// the results are stored
	this -> estimated_state_history.push_back(X_hat);
	this -> residuals.push_back(y_hat);
	this -> estimated_covariance_history.push_back(P_hat);


}



