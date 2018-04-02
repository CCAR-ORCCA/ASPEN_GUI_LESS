#include "ExtendedKalmanFilter.hpp"
#include "DebugFlags.hpp"
#include "System.hpp"
#include "Observer.hpp"
#include <boost/numeric/odeint.hpp>

ExtendedKalmanFilter::ExtendedKalmanFilter(const Args & args) : SequentialFilter(args){
// 
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

