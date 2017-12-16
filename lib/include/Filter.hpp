#ifndef HEADER_FILTER
#define HEADER_FILTER
#include <armadillo>
#include "Args.hpp"
#include <boost/numeric/odeint.hpp>
#include "System.hpp"
#include "Observer.hpp"
#include "FixVectorSize.hpp"

template<typename state_type> class Filter {

public:

	Filter(const Args & args);

	virtual int run(
		unsigned int N_iter,
		const state_type & X0_true,
		const state_type & X_bar_0,
		const std::vector<double> & T_obs,
		const arma::mat & R,
		bool verbose = false) = 0;

	void plot_covariances();
	void plot_residuals();

	// Setters on dynamics, observations
	void set_estimate_dynamics_fun(state_type (*estimate_dynamics_fun)(double, const  state_type & , const Args & args),
		arma::mat (*jacobian_estimate_dynamics_fun)(double, const  state_type & , const Args & args));

	void set_true_dynamics_fun(state_type (*true_dynamics_fun)(double, const  state_type & , const Args & args));

	void set_observations_fun(arma::vec (*observation_fun)(double, const  state_type & , const Args & args),
		arma::mat (*jacobian_observations_fun)(double, const state_type & , const Args & args));


	// Setters on initial state information matrix
	void set_initial_information_matrix(const arma::mat & info_mat_bar_0);

	
	// Getters on results
	std::vector<state_type > get_estimated_state_history() const;
	std::vector<state_type > get_true_state_history() const;
	std::vector<arma::mat > get_estimated_covariance_history() const;

	// Writers
	void write_estimated_state(std::string path_to_estimate) const;
	void write_true_state(std::string path_to_true_state) const;
	virtual void write_estimated_covariance(std::string path_to_covariance) const = 0;
	void write_true_obs(std::string path) const;
	void write_T_obs(const std::vector<double> & T_obs,std::string path) const;
	void write_residuals(std::string path_to_residuals) const;



	

protected:

	// Dynamics
	state_type (*estimate_dynamics_fun)(double, const state_type & , const Args & args) = nullptr;
	arma::mat (*jacobian_estimate_dynamics_fun)(double, const state_type & , const Args & args) = nullptr;

	state_type (*true_dynamics_fun)(double, const state_type & , const Args & args) = nullptr;

	// Observations
	arma::vec (*observation_fun)(double, const state_type & , const Args & args);
	arma::mat (*jacobian_observations_fun)(double, const state_type & , const Args & args);

	// Event function
	arma::vec (*event_function)(double t, const state_type &, const Args &) = nullptr;

	// Output containers
	std::vector< state_type > estimated_state_history;
	std::vector< arma::vec > residuals;
	std::vector< arma::mat > stm_history;
	std::vector< state_type > true_state_history;
	std::vector< arma::mat > estimated_covariance_history;
	std::vector< arma::vec > true_obs_history;


	arma::mat info_mat_bar_0;
	Args args;

};
#endif