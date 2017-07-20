#ifndef HEADER_ICPEXCEPTION
#define HEADER_ICPEXCEPTION

#include <exception>


struct ICPException : public std::exception {
	const char * what () const throw () {
		return "ICP did not converge";
	}
};

#endif