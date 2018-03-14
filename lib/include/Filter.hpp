#ifndef HEADER_FILTER
#define HEADER_FILTER
#include <armadillo>
#include "Args.hpp"

class Filter {

public:

	Filter(const Args & args);

	virtual int run(
		unsigned int N_iter,
		const arma::vec & X0_true,
		const arma::vec & X_bar_0,
		const std::vector<double> & T_obs,
		const arma::mat & R,
		const arma::mat & Q = arma::zeros<arma::mat>(1,1)) = 0;

	void plot_covariances();
	void plot_residuals();

	// Setters on dynamics, observations
	void set_estimate_dynamics_fun(
		arma::vec (*estimate_dynamics_fun)(double, const  arma::vec & , const Args & args),
		arma::mat (*estimate_jacobian_dynamics_fun)(double, const  arma::vec & , const Args & args),
		arma::vec (*true_dynamics_fun)(double, const  arma::vec & , const Args & args) = nullptr);

	void set_observations_fun(
		arma::vec (*estimate_observation_fun)(double, const  arma::vec & , const Args & args),
		arma::mat (*estimate_jacobian_observations_fun)(double, const arma::vec & , const Args & args),
		arma::vec (*true_observation_fun)(double, const arma::vec & , const Args & args) = nullptr);


	// Setters on initial state information matrix
	void set_initial_information_matrix(const arma::mat & info_mat_bar_0);

	// Getters on results
	std::vector<arma::vec > get_estimated_state_history() const;
	std::vector<arma::vec > get_true_state_history() const;
	std::vector<arma::mat > get_estimated_covariance_history() const;

	// Writers
	void write_estimated_state(std::string path_to_estimate) const;
	void write_true_state(std::string path_to_true_state) const;
	void write_estimated_covariance(std::string path_to_covariance) const ;
	void write_true_obs(std::string path) const;
	void write_T_obs(const std::vector<double> & T_obs,std::string path) const;
	void write_residuals(std::string path_to_residuals,const arma::mat & R = arma::zeros<arma::mat>(1)) const;

	virtual void set_gamma_fun(arma::mat (*gamma_fun)(double dt))  = 0;

protected:

	void compute_true_state_history(const arma::vec & X0_true, const std::vector<double> & T_obs);
	void compute_true_observations(const std::vector<double> & T_obs,double mes_noise, double prop_noise = 0);

	// Dynamics
	arma::vec (*estimate_dynamics_fun)(double, const arma::vec & , const Args & args) = nullptr;
	arma::mat (*estimate_jacobian_dynamics_fun)(double, const arma::vec & , const Args & args) = nullptr;
	arma::vec (*true_dynamics_fun)(double, const arma::vec & , const Args & args) = nullptr;

	// Observations
	arma::vec (*estimate_observation_fun)(double, const arma::vec & , const Args & args);
	arma::vec (*true_observation_fun)(double, const arma::vec & , const Args & args);
	arma::mat (*estimate_jacobian_observations_fun)(double, const arma::vec & , const Args & args);

	// Event function
	arma::vec (*event_function)(double t, const arma::vec &, const Args &) = nullptr;

	// Output containers
	std::vector< arma::vec > estimated_state_history;
	std::vector< arma::vec > residuals;
	std::vector< arma::mat > stm_history;
	std::vector< arma::vec > true_state_history;
	std::vector< arma::mat > estimated_covariance_history;
	std::vector< arma::vec > true_obs_history;


	arma::mat info_mat_bar_0;
	Args args;

};
#endif