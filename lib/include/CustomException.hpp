#ifndef HEADER_CUSTOMEXCEPTION
#define HEADER_CUSTOMEXCEPTION

#include <exception>

struct ICPException : public std::exception {
	const char * what () const throw () {
		return "ICP did not converge";
	}
};

struct ICPNoPairsException : public std::exception {
	const char * what () const throw () {
		return "No valid point pairs were found";
	}
};

struct MissingFootpointException : public std::exception {
	const char * what () const throw () {
		return "A footpoint could not be found over the prescribed patch";
	}
};



#endif