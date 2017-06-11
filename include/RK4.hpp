#ifndef HEADER_RK4
#define HEADER_RK4
#include <armadillo>
#include "ShapeModel.hpp"
#include "DynamicAnalyses.hpp"


class RK4 {

public:

	RK4(arma::vec & X0,
	    double t0,
	    double tf,
	    double dt,
	    arma::vec * T,
	    arma::mat * X
	   ) ;




protected:

};


class RK4_attitude : public RK4 {

public:

	RK4_attitude(
	    arma::vec & X0,
	    double t0,
	    double tf,
	    double dt,
	    arma::vec * T,
	    arma::mat * X,
	    arma::mat I,
	    arma::vec (*dXdt)(double, arma::vec,
	                      arma::mat &),
	    bool check_energy_conservation = false
	);


protected:


};

class RK4_orbit : public RK4 {

public:

	RK4_orbit(arma::vec & X0,
	          double t0,
	          double tf,
	          double dt,
	          arma::vec * T,
	          arma::mat * X,
	          double density,
	          ShapeModel * shape_model,
	          FrameGraph * frame_graph,
	          bool check_energy_conservation = false
	         );

protected:


};


#endif
