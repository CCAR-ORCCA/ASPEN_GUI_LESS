#ifndef HEADER_FOOTPOINT
#define HEADER_FOOTPOINT

#include "Element.hpp"


struct Footpoint{
	arma::vec Pbar;
	arma::vec Ptilde;
	double u;
	double v;
	arma::vec n;
	Element * element = nullptr;
};

#endif