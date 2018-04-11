#include "SequentialFilter.hpp"

arma::vec SequentialFilter::compute_residual(double t,const arma::vec & X_hat,
	const arma::vec & Y_true){

	arma::vec y_bar = Y_true - this -> estimate_observation_fun(t,X_hat,this -> args);
	
	return y_bar;

}

void SequentialFilter::apply_SNC(double dt,arma::mat & P_bar,const arma::mat & Q){

	if (this -> gamma_fun != nullptr){
		arma::mat Gamma = this -> gamma_fun(dt);

		if (Q.n_rows == 3){
			P_bar += Gamma * Q * Gamma.t();
		}
		else{
			P_bar.submat(0,0,5,5) += Gamma * Q.submat(0,0,2,2) * Gamma.t();
			P_bar.submat(6,6,11,11) += Gamma * Q.submat(3,3,5,5) * Gamma.t();

		}
	}
}

void SequentialFilter::set_gamma_fun(arma::mat (*gamma_fun)(double dt)){
	this -> gamma_fun = gamma_fun;
}