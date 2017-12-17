#ifndef HEADER_SEQUENTIALFILTER
#define HEADER_SEQUENTIALFILTER

#include <armadillo>
#include "Filter.hpp"

class SequentialFilter : public Filter {

public:

	SequentialFilter(const Args & args) : Filter(args){

	};
	
	virtual int run(
		unsigned int N_iter,
		const arma::vec & X0_true,
		const arma::vec & X_bar_0,
		const std::vector<double> & T_obs,
		const arma::mat & R,
		const arma::mat & Q = arma::zeros<arma::mat>(1,1),
		bool verbose = false) = 0 ;

	virtual void set_gamma_fun(arma::mat (*gamma_fun)(double dt));


protected:

	// What about the true trajectory? should I compute it
	// ahead of time or propagate it along with the estimated state?

	virtual void time_update(double t_now, double t_next,
		arma::vec & X_hat, arma::mat & P_hat) const = 0;
	virtual void measurement_update(double t, arma::vec & X_bar, arma::mat & P_bar,
		const arma::vec & res,const arma::mat & R) const = 0;

	arma::vec compute_residual(double t,const arma::vec & X_hat,
		const arma::vec & Y_true);

	void apply_SNC(double dt,arma::mat & P_bar,const arma::mat & Q);

	arma::mat (*gamma_fun)(double dt) = nullptr;

};
#endif