#ifndef HEADER_OBSERVERS
#define HEADER_OBSERVERS
#include <armadillo>
#include <vector>


namespace Observer {

template <typename state_type> struct push_back_state_and_time{
	std::vector< state_type > & m_states;
	std::vector< double > & m_times;

	push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times ): m_states( states ) , m_times( times ) { }

	void operator()( state_type & x , double t ){

		m_states.push_back( x );
		m_times.push_back( t );
	}
};


template <typename state_type> struct push_back_state {

std::vector< state_type > & m_states;

push_back_state( std::vector< state_type > & states ) : m_states( states )  { 
}

void operator()( state_type & x,  double t){
	m_states.push_back( x );
}
};

template <typename state_type> struct push_back_state_and_energy{

std::vector< state_type > & m_states;
std::vector< double > & m_energy;

push_back_state_and_energy( std::vector< state_type > & states, std::vector< double > & energy ) : m_states(states), m_energy( energy )  { 
}

void operator()( state_type & x,  double t){
	m_states.push_back( x );
	m_energy.push_back( 0.5 * arma::dot(x.rows(3,5),x.rows(3,5))- 1./arma::norm(x.rows(0,2)));
}
};

}

#endif