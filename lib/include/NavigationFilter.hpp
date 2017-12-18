#ifndef HEADER_NAVFILTER
#define HEADER_NAVFILTER

#include <armadillo>
#include "ExtendedKalmanFilter.hpp"
#include "Observations.hpp"
#include "Dynamics.hpp"


class NavigationFilter : public ExtendedKalmanFilter {

public:

	NavigationFilter(const Args & args);
	
	virtual int run(
		unsigned int N_iter,
		const arma::vec & X0_true,
		const arma::vec & X_bar_0,
		arma::vec & X0_true_small_body,
		arma::vec & X0_estimated_small_body,
		const std::vector<double> & T_obs,
		const arma::mat & R = arma::zeros<arma::mat>(1,1),
		const arma::mat & Q = arma::zeros<arma::mat>(1,1),
		bool verbose = false) ;

protected:

	// virtual void time_update(double t_now, double t_next,
	// 	arma::vec & X_hat, arma::mat & P_hat) const;
	// virtual void measurement_update(double t, arma::vec & X_bar, arma::mat & P_bar,
	// 	const arma::vec & res,const arma::mat & R) const;

	void compute_true_small_body_attitude(std::vector<arma::vec> & X_small_body_true,arma::vec & X0_true_small_body,
		const std::vector<double> & T_obs);

	void compute_estimated_small_body_attitude(std::vector<arma::vec> & X_small_body_estimated,arma::vec & X0_estimated_small_body,
		const std::vector<double> & T_obs);
	
};
#endif