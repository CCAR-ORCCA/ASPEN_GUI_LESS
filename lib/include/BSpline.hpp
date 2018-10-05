#ifndef HEADER_BSPLINE
#define HEADER_BSPLINE

#include <ControlPoint.hpp>



class BSpline{

public:

	/**
	Constructor. Will initialize the BSpline as a unit box
	@param l number of unique points per direction
	@param p degree
	*/
	BSpline(int l,int p);


	/**
	Evaluates the (i,j) basis function
	@param t coordinate 0 <= t <= 1
	@param i first index of basis function ( 0 <= i <= N - 1) , N == number of control points
	@param j second index of basis function ( 0 <= j <= N - 1) , N == number of control points
	@param knots vector of knots (should have length of N + 1 - degree)
	@return computed basis function at t
	*/
	static double basis_function( double t, int i, int j,const std::vector<double> & knots);
	
	/**
	Evaluates the BSpline
	@param t first coordinate 0 <= t <= 1
	@param u second coordinate 0 <= t <= 1
	@return computed BSpline at (t,u)
	*/
	arma::vec::fixed<3> evaluate(double t,double u) const;

	/**
	Get i-th knot
	@param i index of knot
	@param return knot
	*/
	double get_knot(int i ) const;


	/**
	Get domain over which the surface is defined
	@param[out] u_min lower bound of domain
	@param[out] u_max upper bound of domain
	*/
	void get_domain(double & u_min,double & u_max) const;

	/**
	Saves control mesh at the provided location
	*/
	void 	save_control_mesh(std::string partial) const;



private:

	double basis_function( double t, int i, int j) const;

	std::vector<ControlPoint> control_points;
	std::vector<std::vector<int> > control_point_indices;

	std::vector<double> knots;

	int n;
	int p;
	int m;


};







#endif