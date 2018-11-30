#ifndef HEADER_OBSERVERS
#define HEADER_OBSERVERS
#include <armadillo>
#include <vector>

#define PUSH_BACK_AUGMENTED_STATE_DEBUG 0
#define PUSH_BACK_ATTITUDE_STATE_DEBUG 0

namespace Observer {

	struct push_back_state_and_time{
		std::vector< arma::vec > & m_states;
		std::vector< double > & m_times;

		push_back_state_and_time( std::vector< arma::vec > &states , std::vector< double > &times ): m_states( states ) , m_times( times ) { }

		void operator()( arma::vec & x , double t ){

			m_states.push_back( x );
			m_times.push_back( t );
		}
	};


	/**
	Observer structure used to push a state into a container. 
	If the state is integrated together with its state transition matrix
	, then the switching of any attitude state possibly present also triggers a switching of the associated state transition matrix partition.
	The presence or absence of stms is inferred from the size of the inegrated state
	compared to the provided number of state components. 
	*/
	struct push_back_state {

		std::vector< arma::vec > & m_states;
		int number_of_states;
		std::vector<int> attitude_state_first_indices;

		/**
		Constructor
		@param[out] states container storing the integrated states
		@param[in] n_states effective number of state components. The integrated states can have n_states, or 
		n_states + n_states * n_states if their associated state transition matrices are integrated alongside them
		@param[in] attitude_state_first_indices list of indices of first component in attitude states. For instance, for an integrated
		that looks like X = {r,r_dot,sigma,omega} where sigma is an attitude set, attitude_state_first_indices should only contain {6} 
		as this is the first index of sigma in X.
		*/
		push_back_state( std::vector< arma::vec > & states , int n_states,std::vector<int> att_state_first_indices = {}) : m_states( states ),
		number_of_states(n_states) ,attitude_state_first_indices(att_state_first_indices) { 

		}

		void operator()( arma::vec & x,  double t){

			if (attitude_state_first_indices.size() > 0){
				
				// Switching matrix, which may or may not be necessary if no switching is detected
				arma::mat Theta;

				// For all the attitude states
				for (auto i : attitude_state_first_indices){


					if (arma::norm(x.subvec(i,i + 2)) > 1){

						const arma::vec::fixed<3> & sigma = x.subvec(i,i + 2);

						x.subvec(i,i + 2) = - sigma / arma::dot(sigma,sigma);

						if (Theta.n_rows == 0 && x.n_rows > number_of_states){

							// There is at least one switching and the stm is being integrated
							// so we can populate Theta
							Theta = arma::eye<arma::mat>(number_of_states,number_of_states);
						}

						if (x.n_rows > number_of_states){
							// The switching matrix partition is populated 
							Theta.submat(i,i,i + 2,i + 2) = 1./(arma::dot(sigma,sigma)) * (2 * sigma * sigma.t() / arma::dot(sigma,sigma) - arma::eye<arma::mat>(3,3));
						}

					}

				}

				if (Theta.n_rows > 0){
					// If the switching matrix was populated, it means that at least one 
					// switching took place so we need to switch the full stm. At this stage, 
					// all switchings have been computed
					x.rows(number_of_states,number_of_states + number_of_states * number_of_states - 1) = arma::vectorise(Theta * arma::reshape(x.rows(number_of_states,number_of_states + number_of_states * number_of_states - 1),number_of_states,number_of_states));
				}
			}

			m_states.push_back( x );

		}
	};


}

#endif