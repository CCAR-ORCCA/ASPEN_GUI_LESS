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
		const arma::mat & Q = arma::zeros<arma::mat>(1,1),
		bool verbose = false) ;

	virtual void set_gamma_fun(arma::mat (*gamma_fun)(double dt));

protected:

	void compute_prefit_residuals(
		const std::vector<double> & T_obs,
		const std::vector<arma::vec> & X_bar,
		std::vector<arma::vec> & y_bar,
		bool & has_converged,
		arma::vec & previous_norm_res_squared,
		const arma::mat & R);


	void compute_reference_state_history(
		const std::vector<double> & T_obs,
		std::vector<arma::vec> & X_bar,
		std::vector<arma::mat> & stm);

	void compute_covariances(const arma::mat & P_hat_0,
		const std::vector<arma::mat> & stm);



};
#endif