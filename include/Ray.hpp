#ifndef HEADER_RAY
#define HEADER_RAY

#include "Facet.hpp"
#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "ShapeModel.hpp"

#include <memory>
#include <armadillo>

class Lidar;
class Facet;
class ShapeModel;

/**
Convenience struct used in the parallelization of the ray casting procedure
*/
struct CompareRanges {
	double range = std::numeric_limits<double>::infinity();
	Facet * hit_facet = nullptr;

};
#pragma omp declare reduction(minimum : struct CompareRanges : omp_out = omp_in.range < omp_out.range ? omp_in : omp_out)

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
	Pointer to hit facet on the true target
	@return pointer to hit facet (set to nullptr if no facet was hit)
	*/
	Facet * get_true_hit_facet() ;

	/**
	Pointer to hit facet on the estimated target
	@return pointer to hit facet (set to nullptr if no facet was hit)
	*/
	Facet * get_computed_hit_facet() ;


	/**
	Value of true range measurement (from pixel to facet)
	@return range measurement (m)
	*/
	double get_true_range() const ;

	/**
	Value of computed range measurement (from pixel to facet)
	@return range measurement (m)
	*/
	double get_computed_range() const ;


	/**
	Sets the true range to the prescribed value
	@param true_range Prescribed true range value
	*/
	void set_true_range(double true_range) ;



	/**
	Sets the computed range to the prescribed value
	@param true_range Prescribed computed range value
	*/
	void set_computed_range(double computed_range) ;


	/**
	Sets the corresponding measurement ray
	to a default state. In particular, the origin and direction
	of the ray are computed in the same reference frame
	as the one corresponding to the Lidar's target coordinates
	@param computed_mes True if the reset ray is the one targeted
	at the computed shape. False if the true shape is targeted
	@param shape_model pointer to shape model about to be flashed by this ray
	*/
	void reset(bool computed_mes, ShapeModel * shape_model) ;


	/**
	Value of range residual (from pixel to facet, true minus computed)
	@return range residual (m)
	*/
	double get_range_residual() const;


	/**
	Sets range residual
	@param res range residual
	**/
	void set_range_residual(double res) ;



	/**
	Return pointer to the unit vector directing the ray,
	expressed in the lidar's reference plane
	@return pointer to ray direction
	*/
	arma::vec * get_direction();

	/**
	Return pointer to the origin of the ray
	expressed in the lidar's reference plane
	@return pointer to ray origin
	*/
	arma::vec * get_origin();

	/**
	Return pointer to the unit vector directing the ray,
	expressed in the target's frame
	@return pointer to ray direction
	*/
	arma::vec * get_direction_target_frame();

	/**
	Return pointer to the origin of the ray
	expressed in the target's frame
	@return pointer to ray origin
	*/
	arma::vec * get_origin_target_frame();

	/**
	Returns the coordinates of the impacted point
	expressed in the instrument frame.
	Throws an exception if this ray has not impacted the target
	@param computed_mes True if the ray is targeted at the computed shape (as opposed to the true shape)
	@return Coordinates of the impacted point expressed in the instrument framme
	*/
	arma::vec get_impact_point(bool computed_mes) const;


	/**
	Cast a ray to the target and searches for intersections inside each of the
	shape model's facets using a greedy search. Not recommended for high resolutins Lidars
	or complex targets
	Sets the $hit_facet and $range members depending on whether an intersect was found:
	- no intersect found: range == oo and hit_facet == nullptr
	- intersect found: hit_facet and range have valid values
	@param computed_mes True if the measurements are collected from the a-priori
	@param shape_model pointer to illuminated shape model
	@return true if the ray hit the target, false otherwise
	*/
	bool brute_force_ray_casting(bool computed_mes ,ShapeModel * shape_model);

	/**
	Cast a ray to a single facet of the target
	Sets the $hit_facet and $range members depending on whether an intersect was found:
	- no intersect found: range == oo and hit_facet == nullptr
	- intersect found: hit_facet and range have valid values
	Rewrites previously found range and intersect if new range is less
	@param computed_mes True if the measurements are collected from the a-priori
	@param hit true if the facet was hit
	*/
	bool single_facet_ray_casting(Facet * facet, bool computed_mes ) ;



	/**
	Accessor to lidar
	@return pointer to parent Lidar
	*/
	Lidar * get_lidar();




protected:

	Lidar * lidar;

	std::shared_ptr<arma::vec> origin;
	std::shared_ptr<arma::vec> direction;

	std::shared_ptr<arma::vec> origin_target_frame;
	std::shared_ptr<arma::vec> direction_target_frame;

	unsigned int row_index;
	unsigned int col_index;

	double true_range;
	Facet * true_hit_facet;

	double computed_range;
	Facet * computed_hit_facet;

	double range_residual;


	bool intersection_inside(arma::vec & H, Facet * facet, double tol = 1e-7) ;


};




#endif