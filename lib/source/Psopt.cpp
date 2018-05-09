#include "Psopt.hpp"
#include "Footpoint.hpp"
#include "Bezier.hpp"


template<class T> Psopt<T>::Psopt(double (*fitfun)(arma::vec, T ,int), const arma::vec & lower_bounds,
const arma::vec & upper_bounds, const unsigned int & population_size,
const unsigned int & iter_max ,
T args) {
	this -> fitfun = fitfun;
	this -> lower_bounds = lower_bounds;
	this -> upper_bounds = upper_bounds;
	this -> population_size = population_size;
	this -> iter_max = iter_max;
	this -> population = arma::zeros <arma::mat> (this -> population_size, this -> lower_bounds.n_rows);
	this -> args = args;
}


template<class T> void Psopt<T>::run(
const bool &  maximize,
const int &  verbose_level,
const std::map<int,std::string> & boundary_conditions,
const double &  max_velocity,
const double &  inertial_weight,
const double &  memory_weight,
const double &  social_weight,
const double &  tolerance,
const int  & convergence_interval) {

	// Check that bounds are consistent
	if (arma::min(this -> upper_bounds - this -> lower_bounds) <= 0){
		throw(std::runtime_error("The lower bounds cannot be less or equal than the upper bounds"));
	}
	if (this -> upper_bounds.size() != this -> lower_bounds.size()){
		throw(std::runtime_error("There must be as many lower bounds ("
			+ std::to_string(this -> lower_bounds.size())  
			+ ") as upper bounds(" 
			+ std::to_string(this -> upper_bounds.size()) + ")"));
	}
	if (this -> upper_bounds.size() == 0){
		throw(std::runtime_error("There cannot be 0 states operated on by the PSO"));
	}

	// Check that boundary conditions are consistent
	std::set<std::string> allowed_boundary_conditions = {"w","c"};

	for (auto iter = boundary_conditions.begin(); iter != boundary_conditions.end(); ++iter){
		if (allowed_boundary_conditions.find(iter -> second) ==  allowed_boundary_conditions.end()){
			throw(std::runtime_error("The boundary condition ' "+ iter -> second + " on the state of index (" + std::to_string(iter -> first) + ") is neither 'w' (wrapping) or 'c' (clamp) "));
		}

	}



	// The population is randomly generated
	for (int state_index = 0; state_index < this -> lower_bounds.n_rows; ++state_index) {
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
	int previous_iter_check = 0;
	double previous_global_best_score;


	if (maximize){
		local_best_score.fill(- arma::datum::inf);
		global_best_score = - arma::datum::inf ;
	}

	else{
		local_best_score.fill(arma::datum::inf);
		global_best_score = arma::datum::inf ;
	}

	previous_global_best_score = global_best_score;



	for (unsigned int iter = 1; iter < this -> iter_max; ++iter)  {

		// The population is updated by adding the velocities to it
		this -> population = this -> population + velocities;

		#pragma omp parallel for 
		for (unsigned int particle = 0; particle < this -> population_size; ++particle) {
			for (unsigned int state_index = 0; state_index < this -> lower_bounds.n_rows; ++state_index) {

				bool wrap;
				bool clamp;

				try  {
					std::string condition = boundary_conditions.at(state_index);
					wrap = ("w" == condition);
					clamp = ("c" == condition);
				} 
				catch(std::out_of_range & e){

					// if no boundary condition was defined for this state, the PSO will fall back to the default
					// clamping condition
					clamp = true;
					wrap = false;
				}


				// Boundary check
				if (this -> population.row(particle)(state_index) > this -> upper_bounds(state_index)) {

					// if this state is flagged as wrappable (think of an angle in [0,2pi]), then it is set to the other bound 
					if(wrap){
						this -> population.row(particle)(state_index) = this -> lower_bounds(state_index); 
						velocities.row(particle)(state_index) = 0;

					}
					else if (clamp){
					// else the state is clamped on the boundary using the 
					// distance to boundary to get within the search interval
						double distance_to_boundary_inside = this -> population.row(particle)(state_index) - this -> upper_bounds(state_index);
						this -> population.row(particle)(state_index) = this -> population.row(particle)(state_index) - distance_to_boundary_inside;
					}
					
				}

				else if (this -> population.row(particle)(state_index) < this -> lower_bounds(state_index)) {
					
					// if this state is flagged as wrappable (think of an angle in [0,2pi]), then it is set to the other bound 
					if(wrap){
						this -> population.row(particle)(state_index) = this -> upper_bounds(state_index); 
						velocities.row(particle)(state_index) = 0;
						
					}
					else if (clamp){
					// else the state is clamped on the boundary using the
					// distance to boundary to get within the search interval
						double distance_to_boundary_inside = this -> lower_bounds(state_index) - this -> population.row(particle)(state_index);

						this -> population.row(particle)(state_index) = this -> population.row(particle)(state_index) + distance_to_boundary_inside;
					}

				}

			}

			// the cost function is evaluated at the particle
			scores(particle) = (* this -> fitfun)(this -> population.row(particle).t(), this -> args,verbose_level);

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
		#pragma omp parallel for 
		for (unsigned int particle = 0; particle < this -> population_size; ++particle) {
			arma::vec random_weights = arma::randu<arma::vec>(2);

			velocities.row(particle) = (inertial_weight * velocities.row(particle) 
				+ random_weights(0) * memory_weight * (local_best.row(particle) - this -> population.row(particle))
				+ random_weights(1) * social_weight * (global_best - this -> population.row(particle)));

			// Velocity dampening
			if (arma::norm(velocities.row(particle)) > max_velocity) {
				velocities.row(particle) = velocities.row(particle) / arma::norm(velocities.row(particle)) * max_velocity;
			}
			
		}

		if (verbose_level > 0) {
			std::cout << std::to_string(iter) << "/" << iter_max << std::endl;
			std::cout << std::endl << "Global best score: " << global_best_score << std::endl;;
			std::cout <<  "Global best at: " << global_best << std::endl;
			std::cout << "Mean velocities: " << arma::mean(velocities,0);
			std::cout << "RMS velocities: " << arma::stddev(velocities,0);
		}

		// Check for convergence


		if (iter > (convergence_interval + previous_iter_check)){

			if (verbose_level > 0) {
				std::cout << "Relative variation in global score since last check: " << std::abs(global_best_score - previous_global_best_score)/std::abs(previous_global_best_score) << std::endl;
			}
			if(std::abs(global_best_score - previous_global_best_score)/std::abs(previous_global_best_score)  < tolerance) {
				break;
			}
			else{
				
				
				previous_global_best_score = global_best_score;
				previous_iter_check = iter;

			}
		}




	}

	if (maximize && verbose_level > 0){
		std::cout << std::endl << "Global Maximum at: " << global_best;
		std::cout << "Global Maximum: " << global_best_score << std::endl;
	}	
	else if (verbose_level > 0){
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




