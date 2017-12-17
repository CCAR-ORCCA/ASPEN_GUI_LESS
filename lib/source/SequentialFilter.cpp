#include "SequentialFilter.hpp"



arma::vec SequentialFilter::compute_residual(double t,const arma::vec & X_hat,
	const arma::vec & Y_true){

	arma::vec y_bar = Y_true - this -> observation_fun(t,X_hat,this -> args);

	return y_bar;

}

void SequentialFilter::apply_SNC(double dt,arma::mat & P_bar,const arma::mat & Q){

	if (this -> gamma_fun != nullptr){
		arma::mat Gamma = this -> gamma_fun(dt);
		P_bar += Gamma * Q * Gamma.t();
	}
}

void SequentialFilter::set_gamma_fun(arma::mat (*gamma_fun)(double dt)){
	this -> gamma_fun = gamma_fun;
}