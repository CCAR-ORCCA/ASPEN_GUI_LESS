#ifndef HEADER_SEQUENTIALFILTER
#define HEADER_SEQUENTIALFILTER

#include <armadillo>
#include "Filter.hpp"

class SequentialFilter : public Filter {

public:

	SequentialFilter(const Args & args);
	
	virtual int run(
		unsigned int N_iter,
		const arma::vec & X0_true,
		const arma::vec & X_bar_0,
		const std::vector<double> & T_obs,
		const arma::mat & R) ;

protected:

	// What about the true trajectory? should I compute it
	// ahead of time or propagate it along with the estimated state?


	virtual void time_update(double t_now, double t_next,arma::vec & X_hat, arma::mat & P_hat) = 0;
	virtual void measurement_update(double t_now,double t_next,arma::vec & X_bar, arma::mat & P_bar) = 0;

	// virtual void compute_reference_state_history(
	// 	const std::vector<double> & T_obs,
	// 	std::vector<arma::vec> & X_bar,
	// 	std::vector<arma::mat> & stm);


};
#endif