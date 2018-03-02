#ifndef HEADER_Bezier
#define HEADER_Bezier

#include <armadillo>
#include "ControlPoint.hpp"
#include "Element.hpp"
#include "Footpoint.hpp"

#include <boost/math/special_functions/factorials.hpp>
#include <memory>
#include <iostream>
#include <RigidBodyKinematics.hpp>

#include <set>
#include <map>

struct NewPoint{
	NewPoint(std::shared_ptr<ControlPoint> point ,
		std::shared_ptr<ControlPoint> end_point ,
		std::tuple<unsigned int, unsigned int, unsigned int> indices_newpoint,
		std::tuple<unsigned int, unsigned int, unsigned int> indices_endpoint){

		this -> point = point;
		this -> end_point= end_point;
		this -> indices_newpoint = indices_newpoint;
		this -> indices_endpoint = indices_endpoint;
	} 

	std::shared_ptr<ControlPoint> point;
	std::shared_ptr<ControlPoint> end_point;
	std::tuple<unsigned int, unsigned int, unsigned int> indices_newpoint;
	std::tuple<unsigned int, unsigned int, unsigned int> indices_endpoint;

};


class ControlPoint;

class Bezier : public Element{

public:

	/**
	Constructor
	@param vertices pointer to vector storing the vertices owned by this facet
	*/
	Bezier( std::vector<std::shared_ptr<ControlPoint > > vertices);

	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbors
	@return Pointer to neighboring facets, plus the calling facet
	*/
	virtual std::set < Element * > get_neighbors(bool all_neighbors) const;


	/**
	Returns pointer to the first vertex owned by $this that is
	neither $v0 and $v1. When $v0 and $v1 are on the same edge,
	this method returns a pointer to the vertex of $this that is not
	on the edge but still owned by $this
	@param v0 Pointer to first vertex to exclude
	@param v1 Pointer to first vertex to exclude
	@return Pointer to the first vertex of $this that is neither $v0 and $v1
	*/
	std::shared_ptr<ControlPoint> vertex_not_on_edge(std::shared_ptr<ControlPoint> v0,
		std::shared_ptr<ControlPoint>v1) const ;

	/**
	Returns patch degree
	@param degree
	*/
	unsigned int get_degree() const;

	/**
	Elevates patch degree by one
	*/
	void elevate_degree();



	/**
	Computes the I1 contribution to the polyhedron center of mass
	@returm I1 com integral 
	*/
	arma::vec I1_cm_int() const;


	/**
	Computes the I2 contribution to the polyhedron center of mass
	@returm I2 com integral 
	*/
	arma::vec I2_cm_int() const;


	/**
	Computes the I3 contribution to the polyhedron center of mass
	@returm I3 com integral 
	*/
	arma::vec I3_cm_int() const;

	
	/**
	Evaluates the bezier patch at the barycentric 
	coordinates (u,v). Note that 0<= u + v <= 1
	@param u first barycentric coordinate
	@param v second barycentric coordinate
	@return point at the surface of the bezier patch
	*/
	arma::vec evaluate(const double u, const double v) const;



	/**
	Evaluates the normal of the bezier patch at the barycentric 
	coordinates (u,v). Note that 0<= u + v <= 1
	@param u first barycentric coordinate
	@param v second barycentric coordinate
	@return point at the surface of the bezier patch
	*/
	arma::vec get_normal(const double u, const double v) ;




	/**
	Adds a new control point to this element from one of its neighbords
	@param element pointer to neighboring element
	@new_point structure storing all the information required to add the provided
	point to this Bezier element
	*/
	void add_point_from_neighbor(Element * element, NewPoint & new_point);


	/**
	Returns the control point given its i and j indices (k = n - i - j)
	@param i first index
	@param j second index
	@return pointer to control point
	*/	
	std::shared_ptr<ControlPoint> get_control_point(unsigned int i, unsigned int j);

	/**
	Returns the tuple of local indices (i,j,k) of a control point within a bezier patch
	@param point pointer to query point
	@return local_indices (i,j,k) 
	*/
	std::tuple<unsigned int, unsigned int,unsigned int> get_local_indices(std::shared_ptr<ControlPoint> point);

	/**
	Returns the coordinates of a control point given its i and j indices (k = n - i - j)
	@param i first index
	@param j second index
	@return coordinats of contorl point
	*/	
	arma::vec get_control_point_coordinates(unsigned int i, unsigned int j) const;


	/**
	Evaluates the partial derivative of Sum( B^n_{i,j,k}C_{ijk}) with respect to (u,v) evaluated 
	at (u,v)
	@param u first coordinate
	@param v second coordinate
	*/
	arma::mat partial_bezier(
		const double u,
		const double v) ;




	/**
	Returns the 3x3 covariance matrix
	tracking the uncertainty in the location of
	a surface point given uncertainty in the patch's control
	points
	@param u mean of first coordinate
	@param v mean of second coordinate
	@param dir direction of ray
	@param P_X covariance on the position 
	of the control points
	@return 3x3 covariance
	*/
	arma::mat covariance_surface_point_deprecated(
		const double u,
		const double v,
		const arma::vec & dir,
		const arma::mat & P_X);





	/**
	Returns the 3x3 covariance matrix
	tracking the uncertainty in the location of
	a surface point given uncertainty in the patch's control
	points using an alternative formulation
	@param u mean of first coordinate
	@param v mean of second coordinate
	@param dir direction of ray
	@param P_X covariance on the position 
	of the control points
	@return 3x3 covariance
	*/
	arma::mat covariance_surface_point(
		const double u,
		const double v,
		const arma::vec & dir,
		const arma::mat & P_X);



	/**
	Computes the patch covariance P_X maximizing the likelihood function 
	associated with the provided footpoint pairs
	@param footpoints vector of footpoints to be used in the training process
	@param P_X trained covariance matrix maximizing p(e0,e1,e2,...,eN; P_X)= Prod(p(e_i;P_X))
	@param diag if true, assumes P_X is diagonal
	*/
	void train_patch_covariance(arma::mat & P_X,const std::vector<Footpoint> & footpoints,bool diag = false);


	/**
	Computes the patch covariance P_X maximizing the likelihood function 
	associated with the provided footpoint pairs
	@param footpoints vector of footpoints to be used in the training process
	*/
	void train_patch_covariance(const std::vector<Footpoint> & footpoints);

	/**
	Computes the patch covariance P_X maximizing the likelihood function 
	associated from the stored footpoint 
	*/
	void train_patch_covariance();


	/**
	Returns the triple product of points i_ = (i,j), j_ = = (k,l) and k_ = = (m,p), e.g Ci_^T(Cj_ x Ck_)
	@param i first index of first point
	@param j second index of first point
	@param k first index of second point
	@param l second index of second point
	@param m first index of third point
	@param p second index of third point
	*/
	double triple_product(const int i ,const int j ,const int k ,const int l ,const int m ,const int p ) const;


	/**
	Computes the quadruple product of points i_ = (i,j), j_ = (k,l), k_ = (m,p), l_ = (q,r)  e.g (Ci_^T Cj_) * (Ck_ x Cl_)
	@param result container storing result of computation
	@param i first index of first point
	@param j second index of first point
	@param k first index of second point
	@param l second index of second point
	@param m first index of third point
	@param p second index of third point
	@param q first index of fourth point
	@param r second index of fourth point
	*/
	void quadruple_product(double * result,const int i ,const int j ,const int k ,const int l ,const int m ,const int p, const int q, const int r ) const;


	// Returns the partial derivative d^2P/(dchi dv)
	arma::mat partial_bezier_dv(
		const double u,
		const double v) ;


	// Returns the partial derivative d^2P/(dchi du)
	arma::mat partial_bezier_du(
		const double u,
		const double v) ;

	/**
	Evaluates the Berstein polynomial
	@param u first barycentric coordinate
	@param v first barycentric coordinate
	@param i first index
	@param j second index
	@param n polynomial order
	@return evaluated bernstein polynomial
	*/
	static double bernstein(
		const double u, 
		const double v,
		const int i,
		const int j,
		const int n) ;



	/**
	Computes the partial derivative of the unit normal vector at the queried point
	with respect to a given control point
	@param u first barycentric coordinate
	@param v first barycentric coordinate
	@param i first index
	@param j second index
	@param n polynomial order
	*/
	arma::mat partial_n_partial_Ck(
		const double u, 
		const double v,
		const int i ,  
		const int j, 
		const int n);

	static double compute_log_likelihood_full_diagonal(arma::vec L, 
		std::pair<const std::vector<Footpoint> * , Bezier * > args);

	static double compute_log_likelihood_block_diagonal(arma::vec L,
		std::pair< const std::vector<Footpoint> * ,std::vector<arma::vec> * > args);

	/**
	Add footpoint to Bezier patch for the covariance training phase
	@param footpoint structure holding Ptilde/Pbar/n/u/v
	*/
	void add_footpoint(Footpoint footpoint);

	/**
	Returns true if the patch has training points already assigned, false otherwise
	@return true if the patch has training points already assigned, false otherwise
	*/
	bool has_footpoints() const;


	/**
	Erases the training data
	*/
	void reset_footpoints();


	/**
	Returns the coefficient alpha_ijk for volume computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double alpha_ijk(const int i, const int j, const int k, const int l, const int m, const int p,const int n);
	
	/**
	Returns the coefficient gamma_ijkl for center of mass computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@param q first index of fourth triplet
	@param r second index of fourth triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double gamma_ijkl(const int i, const int j, const int k, const int l, const int m, const int p,const int q, const int r, const int n);


	/**
	Returns the coefficient kappa_ijkl for inertia of mass computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param m first index of third triplet
	@param p second index of third triplet
	@param q first index of fourth triplet
	@param r second index of fourth triplet
	@param s first index of fifth triplet
	@param t second index of fifth triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double kappa_ijklm(const int i, const int j, const int k, const int l, 
		const int m, const int p,const int q, const int r, 
		const int s, const int t, const int n);


	/**
	Returns the coefficient beta_ijkl for center of mass computation
	@param i first index of first triplet
	@param j second index of first triplet
	@param k first index of second triplet
	@param l second index of second triplet
	@param n patch degree
	@returm computed coefficient
	*/
	static double beta_ijkl( const int i,  const int j,  const int k, const  int l,  const int n);




protected:

	virtual void compute_normal();
	virtual void compute_area();
	virtual void compute_center();



	static double Sa_b(const int a, const int b);
	static double bernstein_coef(const int i , const int j , const int n);


	void construct_index_tables();

	double initialize_covariance(const std::vector<Footpoint> & footpoints,
		std::vector<arma::vec> & v,
		std::vector<arma::vec> & W,
		std::vector<arma::vec> & v_i_norm,
		std::vector<double> & epsilon);


	/**
	Evaluates the quadrature function for surface area computation
	@param u first barycentric coordinate
	@param v second barycentric coordinate (w = 1 - u - v)
	@param g as in A = \int g du dv 
	*/
	double g(double u, double v) const;



	/**
	Returns the partial derivative of the Bernstein polynomial B^n_{i,j,k} evaluated 
	at (u,v)
	@param u first coordinate
	@param v second coordinate
	@param i first index
	@param j second index
	@param n polynomial degree
	@return evaluated partial derivative of the Bernstein polynomial
	*/
	static arma::rowvec partial_bernstein(
		const double u,
		const double v,
		const int i,
		const int j,
		const int n) ;



	/**
	Returns the number of combinations of k items among n.
	Returns 0 if k < 0 or k > n
	@param k subset size
	@param n set size
	@return number of combinations
	*/
	static unsigned int combinations(unsigned int k, unsigned int n);

	/**
	Generates the forward table associating a local index l to the corrsponding triplet (i,j,k) 
	@param n degree
	@return forward look up table
	*/
	static std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > forward_table(unsigned int n);



	/**
	Generates the reverse table associating a local index triplet (i,j,k) to the corresponding
	global index l
	@param n degree
	@return reverse look up table
	*/
	static std::map< std::tuple<unsigned int, unsigned int, unsigned int> ,unsigned int> reverse_table(unsigned int n);



	arma::mat partial_bernstein_dv( 
		const double u, 
		const double v,
		const int i ,  
		const int j, 
		const int n) ;


	arma::mat partial_bernstein_du( 
		const double u, 
		const double v,
		const int i ,  
		const int j, 
		const int n) ;


	unsigned int n;

	std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > forw_table;
	std::map< std::tuple<unsigned int, unsigned int, unsigned int> ,unsigned int> rev_table;



	std::map< Element * , std::vector<NewPoint> > new_points;
	arma::mat P_X;
	std::vector<Footpoint> footpoints;



};
#endif