#ifndef HEADER_RK4
#define HEADER_RK4

#include <armadillo>

#include "Args.hpp"
#include "Interpolator.hpp"



class RK {

public:

	RK(arma::vec  X0,
	   double t0,
	   double tf,
	   double dt,
	   Args * args,
	   std::string title = ""
	  ) ;

	arma::vec * get_T();
	arma::mat * get_X();

protected:

	arma::vec X0;
	double t0;
	double tf;
	double dt;
	arma::vec T;
	arma::mat X;
	arma::vec energy;

	Args * args;
	std::string title;

};

class RK45 : public RK {

public:

	RK45(arma::vec X0,
	     double t0,
	     double tf,
	     double dt,
	     Args * args,
	     std::string title = "",
	     double tol = 1e-4
	    );

	void run(
	    arma::vec (*dXdt)(double, arma::vec, Args *),
	    arma::vec (*event_function)(double t, arma::vec, Args *) = nullptr,
	    bool verbose = false,
	    std::string savepath = "");

protected:
	double tol;

	void propagate(
	    arma::vec (*dXdt)(double, arma::vec, Args *),
	    const double t,
	    const arma::vec & y,
	    arma::vec & y_order_4,
	    arma::vec & y_order_5,
	    arma::mat & K);



};

class RK4 : public RK {

public:

	RK4(
	    arma::vec X0,
	    double t0,
	    double tf,
	    double dt,
	    Args * args,
	    std::string title = ""
	);

	void run(
	    arma::vec (*dXdt)(double, arma::vec , Args * args),
	    arma::vec (*event_function)(double t, arma::vec, Args *) = nullptr,
	    std::string savepath = "");

protected:

};

#endif
