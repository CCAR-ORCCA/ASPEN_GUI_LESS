#ifndef HEADER_DYNAMICANALYSES
#define HEADER_DYNAMICANALYSES

#include "ShapeModelTri.hpp"
#include <armadillo>
#include <boost/progress.hpp>
#include "omp.h"
#include "OMP_flags.hpp"

class DynamicAnalyses {

public:
	/**
	Constructor
	Creates an instance of a DynamicalAnalyses object
	@param shape_model Pointer to shape model for which analysis must be conducted
	*/
	DynamicAnalyses(ShapeModelTri * shape_model);

	/**
	Evaluates the acceleration due to gravity at the provided point using a point mass model
	@param point Array of coordinates at which the acceleration is evaluated
	@param mass Mass of the point mass (kg)
	@return PGM acceleration expressed in the body frame
	*/
	arma::vec point_mass_acceleration(arma::vec & point , double mass) const ;



protected:

	ShapeModelTri * shape_model;

};


#endif