#ifndef HEADER_BATCH_FILTER
#define HEADER_BATCH_FILTER

#include <armadillo>
#include "Filter.hpp"


class BatchFilter : public Filter  {

public:
	BatchFilter(const Args & args);
	
	virtual int run(
		unsigned int N_iter,
		const arma::vec & X0_true,
		const arma::vec & X_bar_0,
		const std::vector<double> & T_obs,
		const arma::mat & R,
		const arma::mat & Q = arma::zeros<arma::mat>(1,1)) ;

	virtual void set_gamma_fun(arma::mat (*gamma_fun)(double dt)){};

protected:

	void compute_prefit_residuals(
		const arma::vec & X_bar,
		arma::vec & y_bar,
		bool & has_converged);


	void compute_covariances(const arma::mat & P_hat_0);



};
#endif