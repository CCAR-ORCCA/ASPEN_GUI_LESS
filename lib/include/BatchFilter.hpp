#ifndef HEADER_BATCH_FILTER
#define HEADER_BATCH_FILTER

#include <armadillo>
#include "Filter.hpp"


template <typename state_type> class BatchFilter : public Filter<state_type>  {

public:

	BatchFilter(const Args & args);
	
	virtual int run(
		unsigned int N_iter,
		const state_type & X0_true,
		const state_type & X_bar_0,
		const std::vector<double> & T_obs,
		const arma::mat & R,
		bool verbose = false) ;

	virtual	void write_estimated_covariance(std::string path_to_covariance) const;


protected:

	void compute_prefit_residuals(
		const std::vector<double> & T_obs,
		const std::vector<state_type> & X_bar,
		std::vector<arma::vec> & y_bar,
		bool & has_converged,
		arma::vec & previous_norm_res_squared,
		const arma::mat & R);


	void compute_reference_state_history(
		const std::vector<double> & T_obs,
		std::vector<state_type> & X_bar,
		std::vector<arma::mat> & stm);

	void compute_covariances(const arma::mat & P_hat_0,
		const std::vector<arma::mat> & stm);


	void compute_true_state_history(const state_type & X0_true, const std::vector<double> & T_obs);
	void compute_true_observations(const std::vector<double> & T_obs,const arma::mat & R);



};
#endif