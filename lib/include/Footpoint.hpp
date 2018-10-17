#ifndef HEADER_FOOTPOINT
#define HEADER_FOOTPOINT

#include "Element.hpp"


struct Footpoint{


	arma::vec::fixed<3> Pbar;
	arma::vec::fixed<3>  Ptilde;
	arma::vec::fixed<3>  ntilde;
	double u;
	double v;
	arma::vec::fixed<3> n;
	int element;
};

#endif