#ifndef HEADER_BSPLINE
#define HEADER_BSPLINE

#include <ControlPoint.hpp>



class BSpline{

public:

	/**
	Constructor. Will initialize the BSpline as a unit box
	@param n_u n_u + 1 == number of basis functions along u direction
	@param n_v n_v + 1 == number of basis functions along v direction
	@param p_u degree of basis functions in u direction
	@param p_v degree of basis functions in v direction
	@param knots_u vector of u knots (if empty, will be made uniform over [0,1])
	@param knots_v vector of v knots (if empty, will be made uniform over [0,1])
	*/
	BSpline(int n_u,int n_v,int p_u,int p_v,
		std::vector<double> knots_u = {},
		std::vector<double> knots_v = {},
		int type = 0);

	/**
	Evaluates the (i,p) basis function
	@param t coordinate 0 <= t <= 1
	@param i index of basis function ( 0 <= i <= N - 1) , N == number of control points
	@param p degree of basis function
	@param knots vector of knots (should have length of N + 1 - degree)
	@return computed basis function at t
	*/
	static double basis_function( double t, int i, int p,const std::vector<double> & knots);
	
	/**
	Evaluates the BSpline
	@param u first coordinate 0 <= t <= 1
	@param v second coordinate 0 <= t <= 1
	@return computed BSpline at (u,v)
	*/
	arma::vec::fixed<3> evaluate(double u,double v) const;


	/**
	Get BSpline domain
	@param[out] u_min minumum of u
	@param[out] v_min minumum of v
	@param[out] u_max maximum of u
	@param[out] v_max maximum of v
	*/
	void get_domain(double & u_min,double & v_min,double & u_max,double & v_max) const;



	/**
	Saves control mesh at the provided location
	*/
	void 	save_control_mesh(std::string partial) const;


	enum Type{
		Opened,Clamped,Closed
	};



private:


	/**
	Determines whether the provided direction is open or clamped
	@param[out] is_open_at_start true if curve is open at beginning
	@param[out] is_open_at_end true if curve is open at end
	@param[in] p curve degree
	@param[in] knots curve knots
	*/
	static void check_if_clamped(bool & is_open_at_start,bool & is_open_at_end,
		int p,const std::vector<double> & knots);

	std::vector<ControlPoint> control_points;
	std::vector<std::vector<int> > control_point_indices;

	std::vector<double> knots_u;
	std::vector<double> knots_v;

	int n_u;
	int n_v;

	int p_u;
	int p_v;

	int m_u;
	int m_v;

	int type;



};







#endif