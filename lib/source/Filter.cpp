#include "Filter.hpp"

Filter::Filter(const Args & args){
	this -> args = args;
}

void Filter::set_estimate_dynamics_fun(arma::vec (*estimate_dynamics_fun)(double, arma::vec , const Args & args),
	arma::mat (*jacobian_estimate_dynamics_fun)(double, arma::vec , const Args & args)){
	this -> estimate_dynamics_fun = estimate_dynamics_fun;
	this -> jacobian_estimate_dynamics_fun = jacobian_estimate_dynamics_fun;
}

void Filter::set_true_dynamics_fun(arma::vec (*true_dynamics_fun)(double, arma::vec , const Args & args)){
	this -> true_dynamics_fun = true_dynamics_fun;
}

void Filter::set_observations_fun(arma::vec (*observation_fun)(double, arma::vec , const Args & args),
	arma::mat (*jacobian_observations_fun)(double, arma::vec , const Args & args)){
	this -> observation_fun = observation_fun;
	this -> jacobian_observations_fun = jacobian_observations_fun;
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


void Filter::compute_true_state_history(const arma::vec & X0_true,
	const std::vector<double> & T_obs){

	this -> true_state_history.push_back(X0_true);

	if (this -> true_dynamics_fun == nullptr){

		// true_state_history  already has one state 
		for (unsigned int i = 1; i < T_obs.size(); ++i){
			this -> true_state_history.push_back(this -> true_state_history[0]);
		}
		
	}

}

void Filter::compute_true_observations(const std::vector<double> & T_obs,const arma::mat & R ){

	this -> true_obs_history.clear();
	arma::mat S =  arma::chol( R, "lower" ) ;
	
	for (unsigned int i = 0; i < T_obs.size(); ++i){

		arma::vec Y = this -> observation_fun(T_obs[i],this -> true_state_history[i],this -> args);
		
		Y += S * arma::randn<arma::vec>( Y.n_rows ) ;

		this -> true_obs_history.push_back(Y);
	}



}



void Filter::write_estimated_state(std::string path_to_estimate) const{

	arma::mat X_hat_history = arma::zeros<arma::mat>(this -> estimated_state_history.at(0).n_rows,this -> estimated_state_history.size());
	for (unsigned int i =0; i < this -> estimated_state_history.size(); ++i ){

		X_hat_history.col(i) = this -> estimated_state_history[i];

	}
	X_hat_history.save(path_to_estimate,arma::raw_ascii);


}




void Filter::write_residuals(std::string path) const{

	arma::mat residuals_mat = arma::zeros<arma::mat>(this -> residuals.at(0).n_rows,this -> residuals.size());
	
	for (unsigned int i =0; i < this -> residuals.size(); ++i ){

		residuals_mat.col(i) = this -> residuals[i];

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

}


