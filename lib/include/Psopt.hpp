#ifndef PSOPT_HPP
#define PSOPT_HPP

#include <armadillo>
#include <map>
#include <cassert>


template <class T> class Psopt {

public:
	
	Psopt(double (*fitfun)(const arma::vec &, T ,int), 
		const arma::vec & lower_bounds,
		const arma::vec & upper_bounds, 
		const unsigned int & population_size,
		const unsigned int & iter_max,
		T args);


	Psopt(double (*fitfun)(const arma::vec &, T ,int), const arma::vec & lower_bounds,
		const arma::vec & upper_bounds, const unsigned int & population_size,
		const unsigned int & iter_max,
		T args,
		const arma::vec & guess);

	arma::vec get_result() const;

	void run(
		const bool&  maximize = false,
		const int & verbose_level = 0,
		const std::map<int,std::string> & boundary_conditions = std::map<int,std::string>(),
		const double & max_velocity = 10,
		const double & inertial_weight = 0.65,
		const double & memory_weight = 2,
		const double & social_weight = 2,
		const double & tolerance = 1e-6,
		const int & convergence_interval = 100.);

	void print_pop();
	void resample(int index_best);


	


protected:
	arma::rowvec result;
	double result_score;
	arma::vec lower_bounds;
	arma::vec guess;
	arma::vec upper_bounds;
	double (*fitfun)(const arma::vec &, T ,int);
	unsigned int population_size;
	unsigned int iter_max;
	arma::mat population;
	T args;

};

#endif

