#ifndef HEADER_RK4
#define HEADER_RK4

#include <armadillo>
#include "ShapeModel.hpp"
#include "DynamicAnalyses.hpp"
#include "FrameGraph.hpp"
#include "Args.hpp"
#include "Interpolator.hpp"



class RK {

public:

	RK(arma::vec  X0,
	   double t0,
	   double tf,
	   double dt,
	   Args * args,
	   bool check_energy_conservation,
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
	bool check_energy_conservation;

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
	     bool check_energy_conservation,
	     std::string title = "",
	     double tol = 1e-3
	    );

	void run(
	    arma::vec (*dXdt)(double, arma::vec, Args *),
	    double (*energy_fun)(double , arma::vec, Args *),
	    arma::vec (*event_function)(double t, arma::vec, Args *) = nullptr,
	    bool verbose = false);

protected:
	double tol;
};

class RK4 : public RK {

public:

	RK4(
	    arma::vec X0,
	    double t0,
	    double tf,
	    double dt,
	    Args * args,
	    bool check_energy_conservation,
	    std::string title = ""
	);

	void run(
	    arma::vec (*dXdt)(double, arma::vec , Args * args),
	    double (*energy_fun)(double, arma::vec , Args * args),
	    arma::vec (*event_function)(double t, arma::vec, Args *) = nullptr);

protected:

};

#endif
