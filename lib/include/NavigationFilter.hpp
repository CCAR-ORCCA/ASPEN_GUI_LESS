#ifndef HEADER_NAVFILTER
#define HEADER_NAVFILTER

#include "ExtendedKalmanFilter.hpp"

class NavigationFilter : public ExtendedKalmanFilter {

public:

	NavigationFilter(const Args & args);
	
	virtual int run(
		unsigned int N_iter,
		const arma::vec & X0_true_augmented,
		const arma::vec & X0_estimated_augmented,
		const std::vector<double> & T_obs,
		const arma::mat & R,
		const arma::mat & Q) ;


protected:

	void compute_true_state(
		std::vector<double> T_obs,
		const arma::vec & X0_true_augmented,
		bool save = false);

	void set_states(const arma::vec & X_hat,arma::vec X_true,unsigned int t);

	
};
#endif