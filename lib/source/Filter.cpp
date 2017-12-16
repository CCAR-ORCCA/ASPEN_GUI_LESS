#include "Filter.hpp"

template<typename state_type> Filter<state_type> ::Filter(const Args & args){
	this -> args = args;
}

template<typename state_type> void Filter<state_type>::set_estimate_dynamics_fun(state_type (*estimate_dynamics_fun)(double, const state_type &, const Args & args),
	arma::mat (*jacobian_estimate_dynamics_fun)(double, const state_type &, const Args & args)){
	this -> estimate_dynamics_fun = estimate_dynamics_fun;
	this -> jacobian_estimate_dynamics_fun = jacobian_estimate_dynamics_fun;
}

template<typename state_type> void Filter<state_type> ::set_true_dynamics_fun(state_type (*true_dynamics_fun)(double, const state_type &, const Args & args)){
	this -> true_dynamics_fun = true_dynamics_fun;
}

template<typename state_type> void Filter<state_type>::set_observations_fun(arma::vec (*observation_fun)(double, const state_type &, const Args & args),
	arma::mat (*jacobian_observations_fun)(double, const state_type &, const Args & args)){
	this -> observation_fun = observation_fun;
	this -> jacobian_observations_fun = jacobian_observations_fun;
}


template<typename state_type> void Filter<state_type>::set_initial_information_matrix(const arma::mat & info_mat_bar_0){
	this -> info_mat_bar_0 = info_mat_bar_0;
}


template<typename state_type> std::vector<state_type > Filter<state_type>::get_estimated_state_history() const{
	return this -> estimated_state_history;
}

template<typename state_type> std::vector<state_type> Filter<state_type>::get_true_state_history() const{
	return this -> true_state_history;
}

template<typename state_type> std::vector<arma::mat> Filter<state_type>::get_estimated_covariance_history() const{
	return this -> estimated_covariance_history;
}




template<typename state_type> void Filter<state_type>::write_estimated_state(std::string path_to_estimate) const{

	arma::mat X_hat_history = arma::zeros<arma::mat>(this -> estimated_state_history.at(0).n_rows,this -> estimated_state_history.size());
	for (unsigned int i =0; i < this -> estimated_state_history.size(); ++i ){

		X_hat_history.col(i) = this -> estimated_state_history[i];

	}
	X_hat_history.save(path_to_estimate,arma::raw_ascii);


}




template<typename state_type> void Filter<state_type>::write_residuals(std::string path) const{

	arma::mat residuals_mat = arma::zeros<arma::mat>(this -> residuals.at(0).n_rows,this -> residuals.size());
	
	for (unsigned int i =0; i < this -> residuals.size(); ++i ){

		residuals_mat.col(i) = this -> residuals[i];

	}

	residuals_mat.save(path,arma::raw_ascii);
}

template<typename state_type> void Filter<state_type>::write_T_obs(const std::vector<double> & T_obs,std::string path) const{

	arma::vec T = arma::zeros<arma::vec>(T_obs.size());
	
	for (unsigned int i =0; i < T_obs.size(); ++i ){

		T(i) = T_obs[i];

	}
	T.save(path,arma::raw_ascii);
}


template<typename state_type> void Filter<state_type>::write_true_obs( std::string path) const {

	arma::mat Y_true_arma = arma::zeros<arma::mat>(this -> true_obs_history.at(0).n_rows,this -> true_obs_history.size());
	
	for (unsigned int i =0; i < this -> true_obs_history.size(); ++i ){

		Y_true_arma.col(i) = this -> true_obs_history[i];

	}
	Y_true_arma.save(path,arma::raw_ascii);

}

template<typename state_type> void Filter<state_type>::write_true_state(std::string path_to_true_state) const {

	arma::mat X_true_history = arma::zeros<arma::mat>(this -> true_state_history.at(0).n_rows,this -> true_state_history.size());
	for (unsigned int i =0; i < this -> true_state_history.size(); ++i ){

		X_true_history.col(i) = this -> true_state_history[i];

	}
	X_true_history.save(path_to_true_state,arma::raw_ascii);

}


// Explicit instantiation
// template class Filter< arma::vec::fixed<2> > ;
// template class Filter< arma::vec::fixed<6> > ;
template class Filter< arma::vec> ;



