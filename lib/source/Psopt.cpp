#include "Psopt.hpp"
#include "Footpoint.hpp"
#include "Bezier.hpp"

template<class T> Psopt<T>::Psopt(double (*fitfun)(arma::vec, T ), const arma::vec & lower_bounds,
const arma::vec & upper_bounds, const unsigned int & population_size,
const unsigned int & iter_max ,
const std::vector<bool> & integer_components, T args) {
	this -> fitfun = fitfun;
	this -> lower_bounds = lower_bounds;
	this -> upper_bounds = upper_bounds;
	this -> population_size = population_size;
	this -> iter_max = iter_max;
	this -> population = arma::zeros <arma::mat> (this -> population_size, this -> lower_bounds.n_rows);
	this -> integer_components = integer_components;
	this -> args = args;

	assert(this -> integer_components.size() == this -> lower_bounds.n_rows);
}


template<class T> Psopt<T>::Psopt(double (*fitfun)(arma::vec, T ), const arma::vec & lower_bounds,
const arma::vec & upper_bounds, const unsigned int & population_size,
const unsigned int & iter_max ,
T args) {
	this -> fitfun = fitfun;
	this -> lower_bounds = lower_bounds;
	this -> upper_bounds = upper_bounds;
	this -> population_size = population_size;
	this -> iter_max = iter_max;
	this -> population = arma::zeros <arma::mat> (this -> population_size, this -> lower_bounds.n_rows);
	this -> integer_components = integer_components;
	this -> args = args;

}


template<class T> void Psopt<T>::run(
bool maximize,
bool pedantic,
double max_velocity,
double inertial_weight,
double memory_weight,
double social_weight,
double tolerance) {


	// The population is randomly generated
	for (unsigned int state_index = 0; state_index < this -> lower_bounds.n_rows; ++state_index) {
		this -> population.col(state_index) = (this -> upper_bounds(state_index)
			- this -> lower_bounds(state_index)) * arma::randu<arma::vec>(this -> population_size)
		+ this -> lower_bounds(state_index);
	}


	// The velocities are generated
	arma::mat velocities = arma::zeros <arma::mat>(this -> population_size, this -> lower_bounds.n_rows);

	// The various structures storing the local/global scores and states are formed
	arma::mat local_best = arma::zeros <arma::mat>(this -> population_size, this -> lower_bounds.n_rows);
	arma::rowvec global_best = arma::zeros < arma::rowvec>(this -> lower_bounds.n_rows);
	arma::vec scores = arma::zeros <arma::vec> (this -> population_size);
	arma::vec local_best_score = arma::vec(this -> population_size);
	
	double global_best_score;


	if (maximize){
		local_best_score.fill(- arma::datum::inf);
		global_best_score = - arma::datum::inf ;
	}

	else{
		local_best_score.fill(arma::datum::inf);
		global_best_score = arma::datum::inf ;
	}


	for (unsigned int iter = 0; iter < this -> iter_max; ++iter)  {
		// The population is updated by adding the velocities to it
		this -> population = this -> population + velocities;

		#pragma omp parallel for 
		for (unsigned int particle = 0; particle < this -> population_size; ++particle) {
			for (unsigned int state_index = 0; state_index < this -> lower_bounds.n_rows; ++state_index) {


				// Boundary check
				if (this -> population.row(particle)(state_index) > this -> upper_bounds(state_index)) {
					while (this -> population.row(particle)(state_index) > this -> upper_bounds(state_index)) {
						this -> population.row(particle)(state_index) = this -> population.row(particle)(state_index) - (this -> upper_bounds(state_index) - this -> lower_bounds(state_index));
					}
				}


				else if (this -> population.row(particle)(state_index) < this -> lower_bounds(state_index)) {
					while (this -> population.row(particle)(state_index) < this -> lower_bounds(state_index)) {
						this -> population.row(particle)(state_index) = this -> population.row(particle)(state_index) + (this -> upper_bounds(state_index) - this -> lower_bounds(state_index));
					}

				}

				// Nearest-integer wrapping
				if (this -> integer_components.size() > 0){
					if (this -> integer_components[state_index] == true) {
						this -> population.row(particle)(state_index) = std::round(this -> population.row(particle)(state_index));
					}
				}
			}

			// the cost function is evaluated at the particle
			scores(particle) = (* this -> fitfun)(this -> population.row(particle).t(), this -> args);

			// The local best is updated if need be
			if (maximize){
				if (scores(particle) > local_best_score(particle)) {
					local_best.row(particle) = this -> population.row(particle);
					local_best_score(particle) = scores(particle);
				}

			}
			else{
				if (scores(particle) < local_best_score(particle)) {
					local_best.row(particle) = this -> population.row(particle);
					local_best_score(particle) = scores(particle);
				}
			}


		}

		// The new global best is found
		unsigned int global_best_index ;

		if (maximize){
			global_best_index = local_best_score.index_max();
		}
		else{
			global_best_index = local_best_score.index_min();
		}
		global_best_score = local_best_score(global_best_index);
		global_best = local_best.row(global_best_index);

		// The velocities for each particle are updated
		for (unsigned int particle = 0; particle < this -> population_size; ++particle) {
			arma::vec random_weights = arma::randu<arma::vec>(2);

			velocities.row(particle) = inertial_weight * velocities.row(particle) + random_weights(0) * memory_weight * (local_best(particle) - this -> population.row(particle))
			+ random_weights(1) * social_weight * (global_best - this -> population.row(particle));

			// Velocity dampening
			if (arma::norm(velocities.row(particle)) > max_velocity) {
				velocities.row(particle) = velocities.row(particle) / arma::norm(velocities.row(particle)) * max_velocity;
			}
		}

		// Check for convergence
		if (iter + 1 == this -> iter_max) {
			if (pedantic){
				std::cout << std::to_string(iter + 1) << "/" << iter_max << std::endl;
			}
			break;
		}
		else if (std::abs(local_best_score.max() - local_best_score.min()) < tolerance) {
			break;
		}





		if (pedantic == true) {
			std::cout << std::to_string(iter + 1) << "/" << iter_max << std::endl;
			std::cout << std::endl << "Global best score: " << global_best_score << std::endl;;
			std::cout <<  "Global best at: " << global_best << std::endl;
			std::cout << "Mean velocities: " << arma::mean(velocities,0);
			std::cout << "RMS velocities: " << arma::stddev(velocities,0);

		}


	}

	if (maximize && pedantic){
		std::cout << std::endl << "Global Maximum at: " << global_best;
		std::cout << "Global Maximum: " << global_best_score << std::endl;
	}	
	else if (pedantic){
		std::cout << std::endl << "Global Minimum at: " << global_best;
		std::cout << "Global Minimum: " << global_best_score << std::endl;
	}
	
	this -> result = global_best;
	this -> result_score = global_best_score;

}




template<class T> void Psopt<T>::print_pop() {
std::cout << this -> population << std::endl;

}

template<class T> arma::vec Psopt<T>::get_result() const{
return this -> result.t();
}

// Explicit declaration
struct RigidTransform;
template class Psopt<std::pair<const std::vector<Footpoint> * ,std::vector<arma::vec> * > >;
template class Psopt<std::pair<const std::vector<Footpoint> * ,Bezier * > >;
template class Psopt<std::vector<RigidTransform> *>;




