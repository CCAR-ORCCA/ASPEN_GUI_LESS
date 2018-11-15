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


	struct push_back_state {

		std::vector< arma::vec > & m_states;

		push_back_state( std::vector< arma::vec > & states ) : m_states( states )  { 
		}

		void operator()( arma::vec & x,  double t){
			m_states.push_back( x );
		}
	};


	struct push_back_attitude_state {

		std::vector< arma::vec > & m_states;

		push_back_attitude_state( std::vector< arma::vec > & states ) : m_states( states )  { 

		}

		void operator()( arma::vec & x,  double t){

			#if PUSH_BACK_ATTITUDE_STATE_DEBUG
			std::cout << "in push_back_attitude_state::operator() at time t == " << t << "\n";
			std::cout << "x == " << x.t() << "\n";
			#endif

			if (arma::norm(x.subvec(0,2)) > 1){

			#if PUSH_BACK_ATTITUDE_STATE_DEBUG
			std::cout << "switching in push_back_attitude_state::operator()\n";
			#endif
				x.subvec(0,2) = - x.subvec(0,2) / arma::dot(x.subvec(0,2),x.subvec(0,2));
				
				if (x.n_rows > 6){
					const arma::vec::fixed<3> & sigma = x.subvec(0,2);
			// Switching matrix
					arma::mat::fixed<6,6> Theta = arma::eye<arma::mat>(6,6);
					Theta.submat(0,0,2,2) = 1./(arma::dot(sigma,sigma)) * (2 * sigma * sigma.t() / arma::dot(sigma,sigma) - arma::eye<arma::mat>(3,3));

			// The stm is switched
					x.subvec(6,6 + 6 * 6 - 1) = arma::vectorise(Theta * arma::reshape(x.subvec(6,6 + 6 * 6 - 1),6,6));
				}
			}



			m_states.push_back( x );

			#if PUSH_BACK_ATTITUDE_STATE_DEBUG
			std::cout << "leaving push_back_attitude_state::operator()\n";
			#endif
		}

	};


	struct push_back_augmented_state {

		std::vector< arma::vec > & m_states;

		push_back_augmented_state( std::vector< arma::vec > & states ) : m_states( states )  { 
		}

		void operator()( arma::vec & x,  double t){

			#if PUSH_BACK_AUGMENTED_STATE_DEBUG
			std::cout << "push_back_augmented_state::operator()\n";
			#endif

			if (arma::norm(x.subvec(6,8)) > 1){

		// The state is switched
				arma::vec sigma = x.subvec(6,8);
				x.subvec(6,8) = - sigma / arma::dot(sigma,sigma);

				if (x.n_rows > 12){
			// Switching matrix
					arma::mat::fixed<12,12> Theta = arma::eye<arma::mat>(12,12);
					Theta.submat(6,6,8,8) = 1./(arma::dot(sigma,sigma)) * (2 * sigma * sigma.t() / arma::dot(sigma,sigma) - arma::eye<arma::mat>(3,3));

			// The stm is switched
					x.rows(12,12 + 12 * 12 - 1) = arma::vectorise(Theta * arma::reshape(x.rows(12,12 + 12 * 12 - 1),12,12));
				}
			}

			m_states.push_back( x );
		}
	};

	struct push_back_augmented_state_no_mrp {

		std::vector< arma::vec > & m_states;

		push_back_augmented_state_no_mrp( std::vector< arma::vec > & states ) : m_states( states )  { 
		}

		void operator()( arma::vec & x,  double t){
			
			m_states.push_back( x );
		}
	};


	struct push_back_state_and_energy{

		std::vector< arma::vec > & m_states;
		std::vector< double > & m_energy;

		push_back_state_and_energy( std::vector< arma::vec > & states, std::vector< double > & energy ) : m_states(states), m_energy( energy )  { 
		}

		void operator()( arma::vec & x,  double t){
			m_states.push_back( x );
			m_energy.push_back( 0.5 * dot(x.subvec(3,5),x.subvec(3,5))- 1./arma::norm(x.subvec(0,2)));
		}
	};

}

#endif