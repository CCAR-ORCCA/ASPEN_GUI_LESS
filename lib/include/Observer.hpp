#ifndef HEADER_OBSERVERS
#define HEADER_OBSERVERS
#include <armadillo>
#include <vector>


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
	if (arma::norm(x.subvec(0,2)) > 1){
		x.subvec(0,2) = - x.subvec(0,2) / arma::dot(x.subvec(0,2),x.subvec(0,2));
	}
	m_states.push_back( x );
}
};



 struct push_back_augmented_state {

std::vector< arma::vec > & m_states;

push_back_augmented_state( std::vector< arma::vec > & states ) : m_states( states )  { 
}

void operator()( arma::vec & x,  double t){

	if (arma::norm(x.subvec(6,8)) > 1){

		// The state is switched
		arma::vec sigma = x.subvec(6,8);
		x.subvec(6,8) = - sigma / arma::dot(sigma,sigma);

		if (x.n_rows > 12){
			// Switching matrix
			arma::mat Theta = arma::eye<arma::mat>(12,12);
			Theta.submat(6,6,8,8) = 1./(arma::dot(sigma,sigma)) * (2 * sigma * sigma.t() / arma::dot(sigma,sigma) - arma::eye<arma::mat>(3,3));

			// The stm is switched
			arma::mat switched_stm = Theta * arma::reshape(x.rows(12,12 + 12 * 12 - 1),12,12);
			x.rows(12,12 + 12 * 12 - 1) = arma::vectorise(switched_stm);
		}
	}
	
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