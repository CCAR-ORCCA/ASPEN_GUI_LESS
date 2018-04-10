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
	Evaluates the gravity acceleration from the provided spherical harmonics 
	coefficients. This function should not be used if 
	the query point lies within the circumscribing sphere surrounding the body .

	The query point coordinates must be expressed in the same frame as the one that 
	was used to evaluate the spherical harmonics coefficients 

	This method was reimplemented by Benjamin Bercovici from the original works
	of Yu Takahashi and Siamak Hesar from CU Boulder.

	For more information, one can read the following references

	- 1: S. V. Bettadpur, "Hotine's geopotential formulation: revisited", Bulletin Geodesique (1995) 69:i35-142
  	- 2: R. A. Werner, "Evaluating Descent and Ascent Trajectories Near Non-Spherical Bodies", Technical Support Package
	- 3: L. E. Cunningham, "On the computation of the spherical harmonic terms needed during the numerical integration of the orbital motion of an artificial satellite"

	@param n_degree degree of the expansion
	@param ref_radius reference radius used in the expansion [L]
	@param mu standard gravitational parameter of the attracting body [L^3/s^2]
	@param pos query point position expressed in the body-fixed frame of the 
	attracting body [L]
	@param Cbar matrix of normalized C coefficients
	@param Sbar matrix of normalized S coefficients
	@return spherical harmonics acceleration evaluated in the body-fixed frame
	*/
	arma::vec spherical_harmo_acc(const unsigned int n_degree,
		const double ref_radius,
		const double  mu,
		arma::vec pos, 
		arma::mat * Cbar,
		arma::mat * Sbar);

	/**
	Computes the jacobian of the dynamics of a rigid body undergoing torque free rotation
	@param attitude attitude set and associated angular velocity
	@param inertia inertia tensor of rigid body
	@return jacobian of dynamics
	*/
	arma::mat attitude_jacobian(arma::vec & attitude ,const arma::mat & inertia) const ;


protected:

	ShapeModelTri * shape_model;

	void GetBnmNormalizedExterior(int n_degree,
		arma::mat & b_bar_real,
		arma::mat & b_bar_imag,
		arma::vec pos,
		double ref_radius);

};


#endif