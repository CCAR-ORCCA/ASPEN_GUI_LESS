#ifndef HEADER_RAY
#define HEADER_RAY

#include <memory>
#include <armadillo>

class Lidar;
class Facet;
class Element;
class ShapeModel;
class Bezier;
class Lidar;

class Ray {

public:

	/**
	Constructor
	@param row_index Index of the row the pixel lies on
	@param col_index Index of the column the pixel lies on
	@param lidar pointer to the instrument owning the ray
	*/
	Ray(unsigned int row_index, unsigned int col_index, Lidar * lidar);

	/**
	Constructor
	@param origin ray origin
	@param direction ray direction (unit vector)
	*/
	Ray(arma::vec::fixed<3> origin, arma::vec::fixed<3> direction);

	/**
	Pointer to hit element on the target
	@return pointer to hit facet (set to nullptr if no facet was hit)
	*/
	int get_hit_element() ;

	/**
	Value of true range measurement (from pixel to facet)
	@return range measurement (m)
	*/
	double get_true_range() const ;

	/**
	Set hit element
	@param hit_element hit element
	*/
	void set_hit_element(int element) ;


	/**
	Sets the true range to the prescribed value
	@param true_range Prescribed true range value
	*/
	void set_true_range(double true_range) ;


	/**
	Sets the corresponding measurement ray
	to a default state, accounting for the attitude of the target
	@param shape_model pointer to shape model about to be flashed by this ray
	*/
	void reset(ShapeModel * shape_model) ;


	/**
	Return pointer to the unit vector directing the ray,
	expressed in the lidar's reference plane
	@return pointer to ray direction
	*/
	const	arma::vec::fixed<3> & get_direction() const;

	/**
	Return pointer to the origin of the ray
	expressed in the lidar's reference plane
	@return pointer to ray origin
	*/
	const	arma::vec::fixed<3> & get_origin() const;

	/**
	Return pointer to the unit vector directing the ray,
	expressed in the target's frame
	@return pointer to ray direction
	*/
	const	arma::vec::fixed<3> & get_direction_target_frame() const;

	/**
	Return pointer to the origin of the ray
	expressed in the target's frame
	@return pointer to ray origin
	*/
	const	arma::vec::fixed<3> & get_origin_target_frame() const;

	/**
	Returns the coordinates of the impacted point
	expressed in the instrument frame.
	Throws an exception if this ray has not impacted the target
	@return Coordinates of the impacted point expressed in the instrument framme
	*/
	arma::vec::fixed<3> get_impact_point() const;

	/**
	Returns the coordinates of the impacted point
	expressed in the target frame.
	Throws an exception if this ray has not impacted the target
	@return Coordinates of the impacted point expressed in the instrument framme
	*/
	arma::vec::fixed<3> get_impact_point_target_frame() const ;



	/**
	Cast a ray to a single facet of the target
	Sets the $hit_facet and $range members depending on whether an intersect was found:
	- no intersect found: range == oo and hit_facet == nullptr
	- intersect found: hit_facet and range have valid values
	Rewrites previously found range and intersect if new range is less
	@param hit true if the facet was hit
	@param store sets the internal state of the ray to reflect the impacted surface if true
	@param outside if true, will only accept the ray if it is cast from the outside of the shape
	*/
	bool single_facet_ray_casting(const Facet & facet,bool store = true,bool outside = true) ;


	/*
	Cast a ray to a single patch of the target
	Sets the $hit_facet and $range members depending on whether an intersect was found:
	- no intersect found: range == oo and hit_facet == nullptr
	- intersect found: hit_facet and range have valid values
	Rewrites previously found range and intersect if new range is less
	@param patch pointer to ray traced patch
	@param u first barycentric coordinate
	@param v second barycentric coordinate
	@param use_KD_impact true if the ray casting must be initialized at the point from the overlaying kd tree
	*/
	
	bool single_patch_ray_casting(const Bezier & patch,double & u, double & v,bool use_KD_impact = true) ;

	int get_guess() const;
	int get_super_element() const;

	void set_guess (int guess);

	arma::vec::fixed<3> get_KD_impact() const;

	/**
	Accessor to lidar
	@return pointer to parent Lidar
	*/
	Lidar * get_lidar();

	/**		
	Get this ray's incidence angle at impact
	@return incidence angle
	*/
	double get_incidence_angle() const;

	void set_impact_coords(const double & u,const double & v);

	void get_impact_coords(double & u_t, double & v_t);

protected:

	Lidar * lidar;

	arma::vec::fixed<3> origin;
	arma::vec::fixed<3> direction;

	arma::vec::fixed<3> origin_target_frame;
	arma::vec::fixed<3> direction_target_frame;

	unsigned int row_index;
	unsigned int col_index;

	int hit_element = -1;
	int super_element = -1;
	int guess;

	double incidence_angle;
	double u;
	double v;
	double true_range =  std::numeric_limits<double>::infinity();

	arma::vec::fixed<3> KD_impact;




	bool intersection_inside(const arma::vec::fixed<3> & H, const Facet & facet, double tol = 1e-7) ;


};




#endif