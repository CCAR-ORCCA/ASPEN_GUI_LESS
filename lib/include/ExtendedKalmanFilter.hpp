#ifndef HEADER_EKF
#define HEADER_EKF

#include "SequentialFilter.hpp"

class ExtendedKalmanFilter : public SequentialFilter {

public:

	ExtendedKalmanFilter(const Args & args);
	
	virtual int run(
		unsigned int N_iter,
		const arma::vec & X0_true,
		const arma::vec & X_bar_0,
		const std::vector<double> & T_obs,
		const arma::mat & R = arma::zeros<arma::mat>(1,1),
		const arma::mat & Q = arma::zeros<arma::mat>(1,1)) {};

protected:

	virtual void time_update(double t_now, double t_next,
		arma::vec & X_hat, arma::mat & P_hat) const;
	virtual void measurement_update(double t, 
		arma::vec & X_bar, 
		arma::mat & P_bar,
		const arma::vec & res,
		const arma::mat & R,
		bool & done_iterating,
		double & previous_mahalanobis_distance) const;

};
#endif