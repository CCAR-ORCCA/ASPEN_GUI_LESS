#ifndef PSOPT_HPP
#define PSOPT_HPP

#include <armadillo>
#include <map>
#include <cassert>


template <class T> class Psopt {

public:
	// constructor
	Psopt(double (*fitfun)(arma::vec, T ), const arma::vec & lower_bounds,
		const arma::vec & upper_bounds, const unsigned int & population_size,
		const unsigned int & iter_max,
		const std::vector<bool> & integer_components, T  args);

	Psopt(double (*fitfun)(arma::vec, T ), const arma::vec & lower_bounds,
		const arma::vec & upper_bounds, const unsigned int & population_size,
		const unsigned int & iter_max,
		T args);

	arma::vec get_result() const;

	void run(
		bool maximize = false,
		bool pedantic = false,
		bool resample = false,
		double max_velocity = 10,
		double inertial_weight = 0.65,
		double memory_weight = 2,
		double social_weight = 2,
		double tolerance = 1e-3);

	void print_pop();


	


protected:
	arma::rowvec result;
	double result_score;
	arma::vec lower_bounds;
	arma::vec upper_bounds;
	double (*fitfun)(arma::vec, T );
	unsigned int population_size;
	unsigned int iter_max;
	arma::mat population;
	std::vector<bool> integer_components;
	T args;


	void resample(int global_best_index);


};

#endif

