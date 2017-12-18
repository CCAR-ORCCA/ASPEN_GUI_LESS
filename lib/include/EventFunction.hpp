#ifndef HEADER_EVENTFUNCTION
#define HEADER_EVENTFUNCTION

#include <armadillo>
#include "Args.hpp"


namespace EventFunction{

	
	arma::vec event_function_mrp_omega(double t , arma::vec X, Args * args);
	arma::vec event_function_mrp(double t , arma::vec X, Args * args);

}



#endif