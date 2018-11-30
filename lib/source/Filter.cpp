#include "Filter.hpp"
#include "DebugFlags.hpp"
#include "FixVectorSize.hpp"

#include "Observer.hpp"
#include <boost/numeric/odeint.hpp>


Filter ::Filter(const Args & args){
	this -> args = args;
	this -> true_dynamics_system = SystemDynamics(this -> args);
	this -> estimated_dynamics_system = SystemDynamics(this -> args);

}




void Filter::set_observations_function_estimate(arma::vec (*estimate_observation_fun)(double, const arma::vec &, const Args & args)){
	this -> estimate_observation_fun = estimate_observation_fun; 
}

void Filter::set_jacobian_observations_function_estimate(arma::mat (*estimate_jacobian_observations_fun)(double, const arma::vec &, const Args & args)){
	this -> estimate_jacobian_observations_fun = estimate_jacobian_observations_fun; 
}

void Filter::set_true_observations_fun(arma::vec (*true_observation_fun)(double, const arma::vec &, const Args & args)){
	this -> true_observation_fun = true_observation_fun; 
}





void Filter::set_observations_funs(
	arma::vec (*estimate_observation_fun)(double, const arma::vec &, const Args & args),
	arma::mat (*estimate_jacobian_observations_fun)(double, const arma::vec &, const Args & args),
	arma::vec (*true_observation_fun)(double, const arma::vec &, const Args & args)){
	this -> estimate_observation_fun = estimate_observation_fun;
	this -> estimate_jacobian_observations_fun = estimate_jacobian_observations_fun;

	if (true_observation_fun == nullptr){
		this -> true_observation_fun = estimate_observation_fun;
	}
	else{
		this -> true_observation_fun = true_observation_fun;
	}

}


void Filter::set_initial_information_matrix(const arma::mat & info_mat_bar_0){
	this -> info_mat_bar_0 = info_mat_bar_0;
}


std::vector<arma::vec > Filter::get_estimated_state_history() const{
	return this -> estimated_state_history;
}

std::vector<arma::vec> Filter::get_true_state_history() const{
	return this -> true_state_history;
}

std::vector<arma::mat> Filter::get_estimated_covariance_history() const{
	return this -> estimated_covariance_history;
}




void Filter::write_estimated_state(std::string path_to_estimate) const{

	arma::mat X_hat_history = arma::zeros<arma::mat>(this -> estimated_state_history.at(0).n_rows,this -> estimated_state_history.size());
	for (unsigned int i =0; i < this -> estimated_state_history.size(); ++i ){

		X_hat_history.col(i) = this -> estimated_state_history[i];

	}
	X_hat_history.save(path_to_estimate,arma::raw_ascii);


}


void Filter::set_true_dynamics_system(SystemDynamics true_dynamics_system){
	this -> true_dynamics_system= true_dynamics_system;
}

void Filter::set_estimated_dynamics_system(SystemDynamics estimated_dynamics_system){
	this -> estimated_dynamics_system= estimated_dynamics_system;
}


void Filter::write_residuals(std::string path,const arma::mat & R) const{

	arma::mat residuals_mat = arma::zeros<arma::mat>(this -> residuals.at(0).n_rows,this -> residuals.size());

	for (unsigned int i =0; i < this -> residuals.size(); ++i ){

		if (R.max() != 0){
			residuals_mat.col(i) = this -> residuals[i] / arma::sqrt(R.diag());
		}
		else{
			residuals_mat.col(i) = this -> residuals[i];
		}

	}

	residuals_mat.save(path,arma::raw_ascii);
}

void Filter::write_T_obs(const std::vector<double> & T_obs,std::string path) const{

	arma::vec T = arma::zeros<arma::vec>(T_obs.size());

	for (unsigned int i =0; i < T_obs.size(); ++i ){

		T(i) = T_obs[i];

	}
	T.save(path,arma::raw_ascii);
}


void Filter::write_true_obs( std::string path) const {

	arma::mat Y_true_arma = arma::zeros<arma::mat>(this -> true_obs_history.at(0).n_rows,this -> true_obs_history.size());

	for (unsigned int i =0; i < this -> true_obs_history.size(); ++i ){

		Y_true_arma.col(i) = this -> true_obs_history[i];

	}
	Y_true_arma.save(path,arma::raw_ascii);

}

void Filter::write_true_state(std::string path_to_true_state) const {

	arma::mat X_true_history = arma::zeros<arma::mat>(this -> true_state_history.at(0).n_rows,this -> true_state_history.size());
	for (unsigned int i =0; i < this -> true_state_history.size(); ++i ){

		X_true_history.col(i) = this -> true_state_history[i];

	}
	X_true_history.save(path_to_true_state,arma::raw_ascii);

}

void Filter::write_estimated_covariance(std::string path_to_covariance) const{

	unsigned int state_dim = this -> true_state_history.at(0).n_rows;

	arma::mat P_hat_history = arma::zeros<arma::mat>(state_dim,state_dim * this -> true_state_history.size());
	for (unsigned int i =0 ; i < this -> true_state_history.size(); ++i ){

		P_hat_history.cols(i * state_dim,i * state_dim + state_dim - 1) = this -> estimated_covariance_history[i];

	}
	P_hat_history.save(path_to_covariance,arma::raw_ascii);

}


void Filter::compute_true_state_history(const arma::vec & X0_true,
	const std::vector<double> & T_obs){

		//
		// Odeint is called here. the state is propagated
		// along with the state transition matrices
		//

	unsigned int N_true = X0_true.n_rows;
	arma::vec X0_true_copy(X0_true);


	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 , 1.0e-16 );

	auto tbegin = T_obs.begin();
	auto tend = T_obs.end();

	boost::numeric::odeint::integrate_times(stepper, this -> true_dynamics_system, X0_true_copy, tbegin, tend,1e-10,
		Observer::push_back_state(this -> true_state_history,
			this -> true_dynamics_system.get_number_of_states(),
			this -> true_dynamics_system.get_attitude_state_first_indices()));




}

void Filter::compute_true_observations(const std::vector<double> & T_obs,
	double mes_noise_sd,
	double prop_noise_sd ){

	this -> true_obs_history.clear();
	assert(T_obs.size() == 1);

	arma::vec Y = this -> true_observation_fun(T_obs[0],this -> true_state_history[0],this -> args);

	Y += (mes_noise_sd + prop_noise_sd * Y) % arma::randn<arma::vec>( Y.n_rows ) ;

	this -> true_obs_history.push_back(Y);

}



