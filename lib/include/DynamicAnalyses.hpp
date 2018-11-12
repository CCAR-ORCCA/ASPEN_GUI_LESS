#ifndef HEADER_DYNAMICANALYSES
#define HEADER_DYNAMICANALYSES

#include <armadillo>
#include <boost/progress.hpp>
#include "omp.h"
#include "OMP_flags.hpp"


template <class PointType> 
class ShapeModel;


template <class PointType> 
class ShapeModelTri;

class ControlPoint;
class DynamicAnalyses {

public:
	/**
	Constructor
	Creates an instance of a DynamicalAnalyses object
	@param shape_model Pointer to shape model for which analysis must be conducted
	*/
	DynamicAnalyses(ShapeModelTri<ControlPoint> * shape_model);

	/**
	Evaluates the acceleration due to gravity at the provided point using a point mass model
	@param point Array of coordinates at which the acceleration is evaluated
	@param mass Mass of the point mass (kg)
	@return point mass acceleration expressed in the body frame
	*/
	arma::vec point_mass_acceleration(arma::vec & point , double mass) const ;

	/**
	Evaluates the gravity gradient matrix at the provided point using a point mass model
	@param point Array of coordinates at which the acceleration is evaluated
	@param mass Mass of the point mass (kg)
	@return gravity gradient expressed in the inertial frame of reference
	*/
	arma::mat point_mass_jacobian(arma::vec & point , double mass) const ;

	

	/**
	Computes the jacobian of the dynamics of a rigid body undergoing torque free rotation
	@param attitude attitude set and associated angular velocity
	@param inertia inertia tensor of rigid body
	@return jacobian of dynamics
	*/
	arma::mat attitude_jacobian(arma::vec & attitude ,const arma::mat & inertia) const ;


protected:

	ShapeModelTri<ControlPoint> * shape_model;

	void GetBnmNormalizedExterior(int n_degree,
		arma::mat & b_bar_real,
		arma::mat & b_bar_imag,
		arma::vec pos,
		double ref_radius);

};


#endif