#ifndef HEADER_SYSTEMDYNAMICS
#define HEADER_SYSTEMDYNAMICS

#include <armadillo>
#include "FixVectorSize.hpp"
#include "Args.hpp"

#define SYSTEMDYNAMICS_DEBUG 0

class SystemDynamics {
public:

	/**
	Default Constructor
	*/
	SystemDynamics() {

	}
	
	/**
	Constructor
	@param args structure of arguments that may be necessary for the evaluation of the dynamics and that are not
	stored in the integrated state
	*/
	SystemDynamics(Args args) {
		this -> args = args ;
	}

	/**
	Adds a new state to the state manager
	@param name name of the new state (exemple: "position", "SRP_coef",...)
	@param size number of components in the considered state
	@param is_mrp_state if true, will populate an internal container allowing proper 
	handling of mrp switches
	*/
	void add_next_state(std::string name,unsigned int size,bool is_mrp_state){

		if (size == 0){
			throw(std::runtime_error("A state must have at least one component"));
		}

		arma::uvec indices(size);
		
		for (unsigned int i = 0; i < size; ++i){
			indices(i) = this -> number_of_states + i;
		}

		if (is_mrp_state){
			this -> attitude_state_first_indices.push_back(indices(0));
		}

		this -> number_of_states += size;

		state_indices[name] = indices;

	}

	/**
	Adds one dynamics function to the list of functions to evaluate to compute the time derivative
	of the corresponding state 
	@param state name of the state whose time derivative is to be computed. This state must have been added to the state manager
	@param effect pointer to the function to be added to the list of functions to evaluate to compute the derivative
	of the corresponding state 	
	@param input_states ordered list of strings identifying the states that are required as inputs to the provided function pointer. These states
	must have been added to the state manager
	*/
	void add_dynamics(std::string state,arma::vec (*effect)(double, const arma::vec &  , const Args & args),std::vector<std::string> input_states){
		

		for (auto input_state : input_states){

			if (this -> state_indices.find(input_state) == this -> state_indices.end()){
				throw(std::runtime_error("in StateDynamics::add_dynamics, input state '" + input_state + "' does not exist"));
			}

		}



		if (this -> state_indices.find(state) != this -> state_indices.end()){
			this -> dynamics[state].push_back(std::make_pair(effect,input_states));
		}
		else{
			throw(std::runtime_error("in StateDynamics::add_dynamics, state '" + state + "' does not exist"));
		}
	}



	/**
	Adds one jacobian function to the list of functions adding up to compute the partial derivative of the time derivative
	of the corresponding state with respect to another existing state. For instance, for a full state
	X = {position, velocity}, the time derivative reads X_dot == {velocity,acceleration}. In this case, adding a jacobian function to the partial derivative 
	of the velocity with respect to the position would require calling `add_jacobian` with state == "position" and wr_state == "position"
	@param state name of the state whose time derivative is partially differentiated. This state must have been added to the state manager. 
	@param wr_state the state with respect to which the partial derivative is to be computed
	@param jacobian pointer to the function to be added to the list of functions to evaluate to compute the partial derivative ot the time derivative
	of the corresponding state 	
	@param input_states ordered list of strings identifying the states that are required as inputs to the provided function pointer. These states
	must have been added to the state manager
	*/
	void add_jacobian(std::string state,std::string wr_state, arma::mat (*jacobian)(double, const arma::vec &  , const Args & args),std::vector<std::string> input_states){
		

		for (auto input_state : input_states){

			if (this -> state_indices.find(input_state) == this -> state_indices.end()){
				throw(std::runtime_error("in StateDynamics::add_jacobian, input state '" + input_state + "' does not exist"));
			}

		}

		if (this -> state_indices.find(wr_state) == this -> state_indices.end()){
			throw(std::runtime_error("in StateDynamics::add_jacobian, differenting state '" + wr_state + "' does not exist"));

		}

		if (this -> state_indices.find(state) != this -> state_indices.end() ){
			this -> jacobians[state][wr_state].push_back(std::make_pair(jacobian,input_states));
		}
		else{
			throw(std::runtime_error("in StateDynamics::add_jacobian, differentiated state '" + state + "' does not exist"));
		}



	}


	/**
	Returns the list of indices of potential attitude states. For instance, for an integrated
		that looks like X = {r,r_dot,sigma,omega} where sigma is an attitude set, attitude_state_first_indices should only contain {6} 
		as this is the first index of sigma in X.
	*/
	std::vector<int> get_attitude_state_first_indices() const{
		return this -> attitude_state_first_indices;
	}






	void operator() (const arma::vec & X , arma::vec & dxdt , const double t ){

		arma::vec derivative = arma::zeros<arma::vec>(X.n_rows);


		

		this -> evaluate_state_derivative(X,derivative,t);


		if (jacobians.size() > 0){
			this -> evaluate_stm_derivative(X,derivative,t);
		}

		dxdt = derivative;


	}



	int get_number_of_states() const{
		return this -> number_of_states;
	}


	void evaluate_state_derivative(const arma::vec & X , arma::vec & derivative , const double t ) const{

	// For all the states for which time derivatives must be computed
		for (auto state_dynamics_iter = this -> dynamics.begin(); state_dynamics_iter != this -> dynamics.end(); ++state_dynamics_iter){

			const auto & state_indices_in_X = this -> state_indices.at(state_dynamics_iter -> first);

			const auto & state_dynamics = state_dynamics_iter -> second;

			// For all the dynamics acting on this state
			for (unsigned int k = 0 ; k < state_dynamics.size(); ++k){


				int inputs = 0;
				std::vector<arma::uvec> input_indices_vector;

				// For all the inputs needed by this particular function
				for (unsigned int j = 0; j < state_dynamics[k].second.size(); ++j){

					arma::uvec input_indices = arma::regspace<arma::uvec>(inputs,inputs + static_cast<int>(this -> state_indices.at(state_dynamics[k].second[j]).size()) -1 );
					inputs += this -> state_indices.at(state_dynamics[k].second[j]).size();
					input_indices_vector.push_back(input_indices);
				}

				arma::vec X_input = arma::zeros<arma::vec>(inputs);

				// For all the inputs needed by this particular function

				for (unsigned int j = 0; j < state_dynamics[k].second.size(); ++j){

					const auto & indices_in_input = input_indices_vector[j];
					const auto & indices_in_X = this -> state_indices.at(state_dynamics[k].second[j]);


					X_input.subvec(indices_in_input(0),indices_in_input(static_cast<int>(indices_in_input.n_rows) - 1)) = X.subvec(indices_in_X(0),indices_in_X(static_cast<int>(indices_in_X.n_rows) - 1));
				}

				// The partition of the full state time derivative corresponding to the considered state is incremented
				derivative.subvec(state_indices_in_X(0),state_indices_in_X(static_cast<int>(state_indices_in_X.n_rows) - 1)) += state_dynamics[k].first(t,X_input,this -> args);

			}
		}
	}


	void evaluate_stm_derivative(const arma::vec & X , arma::vec & derivative , const double t) const{

			#if SYSTEMDYNAMICS_DEBUG
		std::cout << "in jacobians. States: " << this -> number_of_states << "\n";
			#endif

		arma::mat A = this -> compute_A_matrix(X,t);


		arma::mat Phi = arma::reshape(X.subvec(number_of_states,
			number_of_states + number_of_states * number_of_states - 1), 
		number_of_states, number_of_states );


		derivative.subvec(number_of_states,
			number_of_states + number_of_states * number_of_states - 1) = arma::vectorise(A * Phi);
	}

	arma::mat compute_A_matrix(const arma::vec & X , const double t) const{


		arma::mat A = arma::zeros<arma::mat>(this -> number_of_states,this -> number_of_states);

			// For all the states time derivatives for which a partial derivative must be computed
		for (auto state_jacobians_iter = jacobians.begin(); state_jacobians_iter !=jacobians.end() ; ++state_jacobians_iter){


			const auto & state_indices_in_X = this -> state_indices.at(state_jacobians_iter -> first);

			const auto & state_jacobians = state_jacobians_iter -> second;


				#if SYSTEMDYNAMICS_DEBUG
			std::cout << state_jacobians_iter -> first << std::endl;
				#endif

				// For all the states with respect to which a partial derivative will be taken
			for (auto differentiating_state_iter = state_jacobians.begin(); differentiating_state_iter != state_jacobians.end(); ++differentiating_state_iter){


					#if SYSTEMDYNAMICS_DEBUG
				std::cout << "\t " << differentiating_state_iter -> first << std::endl;
					#endif

				const auto & differentiating_state_indices_in_X = this -> state_indices.at(differentiating_state_iter -> first);

				const auto & differentiating_state_jacobians = differentiating_state_iter -> second;

					// For all the jacobians needed to compute d(*state_jacobians_iter) / d(*differentiating_state_iter)
				for (unsigned int k = 0 ; k < differentiating_state_jacobians.size(); ++k){

					int inputs = 0;
					std::vector<arma::uvec> input_indices_vector;

						// For all the states taken as inputs to this particular jacobian function
					for (unsigned int j = 0; j < differentiating_state_jacobians[k].second.size(); ++j){

						arma::uvec input_indices = arma::regspace<arma::uvec>(inputs,inputs + static_cast<int>(state_indices.at(differentiating_state_jacobians[k].second[j]).size()) - 1);
						inputs += this -> state_indices.at(differentiating_state_jacobians[k].second[j]).size();
						input_indices_vector.push_back(input_indices);
					}

					arma::vec X_input = arma::zeros<arma::vec>(inputs);

						// For all the states taken as inputs to this particular jacobian function
					for (unsigned int j = 0; j < differentiating_state_jacobians[k].second.size(); ++j){

						const auto & indices_in_input = input_indices_vector[j];
						const auto & indices_in_X = this -> state_indices.at(differentiating_state_jacobians[k].second[j]);


						X_input.subvec(indices_in_input(0),indices_in_input(static_cast<int>(indices_in_input.n_rows) - 1)) = X.subvec(indices_in_X(0),indices_in_X(static_cast<int>(indices_in_X.n_rows) - 1));
					}

						// The partition corresponding to d(*state_jacobians_iter) / d(*differentiating_state_iter)
						// is incremented with the contribution from the particular jacobian function
					A.submat(state_indices_in_X(0),
						differentiating_state_indices_in_X(0),
						state_indices_in_X(static_cast<int>(state_indices_in_X.n_rows) - 1),
						differentiating_state_indices_in_X(static_cast<int>(differentiating_state_indices_in_X.n_rows) - 1)) += differentiating_state_jacobians[k].first(t,X_input,this -> args);

				}

			}

		}

		return A;




	}


protected:







	std::map<std::string,arma::uvec> state_indices;
	std::map<std::string,std::vector< std::pair< arma::vec (*)(double, const arma::vec &  , const Args & args),std::vector<std::string > > > > dynamics;
	std::map<std::string,std::map<std::string,std::vector< std::pair< arma::mat (*)(double, const arma::vec &  , const Args & args),std::vector<std::string > > > > > jacobians;

	std::vector<int> attitude_state_first_indices;
	
	Args args;
	int number_of_states = 0;
};



	#endif