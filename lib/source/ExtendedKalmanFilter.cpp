#include "ExtendedKalmanFilter.hpp"
#include "Observer.hpp"
#include <boost/numeric/odeint.hpp>


#define EKF_DEBUG 0


ExtendedKalmanFilter::ExtendedKalmanFilter(const Args & args) : SequentialFilter(args){
// 
}


void ExtendedKalmanFilter::time_update(double t_now, double t_next,
	arma::vec & X_hat, arma::mat & P_hat) const{

	#if EKF_DEBUG
	std::cout << "in ExtendedKalmanFilter::time_update\n";
	std::cout << "state size: " << X_hat.n_rows << std::endl;
	#endif


	unsigned int N_est = X_hat.n_rows;


	

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

	#if EKF_DEBUG
	std::cout << "integrating dynamics\n";
	#endif
	
	boost::numeric::odeint::integrate_times(stepper, this -> estimated_dynamics_system, x, tbegin, tend,1e-10,
		Observer::push_back_state(augmented_state_history,
			this -> estimated_dynamics_system.get_number_of_states(),
			this -> estimated_dynamics_system.get_attitude_state_first_indices()
			));

	if (augmented_state_history.size() != 2){
		throw(std::runtime_error("augmented_state_history should have two elements only"));
	}

	for (unsigned int i = 0; i < N_est; ++i){
		X_hat(i) = augmented_state_history[1](i);
	}

	#if EKF_DEBUG
	std::cout << "building stm\n";
	#endif
	
	arma::mat stm = arma::reshape(
		augmented_state_history[1].rows(N_est,N_est + N_est * N_est - 1),
		N_est,N_est);

	

	P_hat = stm * P_hat * stm.t();

}

void ExtendedKalmanFilter::measurement_update(double t,
	arma::vec & X_bar,
	arma::mat & P_bar,
	const arma::vec & res,
	const arma::mat & R,
	bool & done_iterating,
	double & previous_mahalanobis_distance) const{


	std::cout << "-- EKF measurement update\n";
	auto H = this -> estimate_jacobian_observations_fun(t, X_bar , this -> args);

	// The Kalman gain is computed
	arma::mat K = P_bar * H.t() * arma::inv(H * P_bar * H.t() + R);

	std::cout << "--- State covariance before update\n";
	std::cout << P_bar << std::endl;

	std::cout << "--- Kalman gain :\n";
	std::cout << K << std::endl;

	std::cout << "--- Residuals :\n";
	std::cout << res << std::endl;

	std::cout << "--- Innovation :\n";
	std::cout << K * res << std::endl;

	// The innovation is added to the state
	X_bar = X_bar + K * res;

	// Consistency test to see if the filter has converged
	auto I = arma::eye<arma::mat>(X_bar.n_rows,X_bar.n_rows);
	arma::mat P_hat = (I - K * H) * P_bar * (I - K * H).t() + K * R * K.t();
	double mahalanobis_distance = arma::dot(K * res,arma::solve(P_hat,K * res));

	double mahalanobis_distance_variation = (previous_mahalanobis_distance - mahalanobis_distance)/mahalanobis_distance * 100 ; 

	std::cout << "--- Mahalanobis distance: \n" << mahalanobis_distance << " \n";
	std::cout << "--- Variation of Mahalanobis distance: \n" << mahalanobis_distance_variation << " \n";

	if (mahalanobis_distance_variation < 0){
		done_iterating = true;
	}
	else{
		previous_mahalanobis_distance = mahalanobis_distance;
	}

	// The covariance is updated
	if (done_iterating){
		P_bar = P_hat;
		std::cout << "--- State covariance after update\n";
		std::cout << P_bar << std::endl;

	}



}

