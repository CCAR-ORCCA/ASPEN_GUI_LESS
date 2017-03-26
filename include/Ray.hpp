#ifndef HEADER_RAY
#define HEADER_RAY

#include "Facet.hpp"
#include "Lidar.hpp"

#include <memory>
#include <armadillo>

class Lidar;
class Facet;
class ShapeModel;

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
	Pointer to hit facet
	@return pointer to hit facet (set to nullptr if no facet was hit)
	*/
	Facet * get_hit_facet() ;

	/**
	Value of range measurement (from pixel to facet)
	@return range measurement (m)
	*/
	double get_range() const ;

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
	Cast a ray to the target and searches for intersections inside each of the
	shape model's facets.
	Sets the $hit_facet and $range members depending on whether an intersect was found:
		- no intersect found: range == oo and hit_facet == nullptr
		- intersect found: hit_facet and range have valid values
	@param computed_mes True if the measurements are collected from the a-priori
	*/
	void brute_force_ray_casting(bool computed_mes = false);


protected:
	Lidar * lidar;
	std::shared_ptr<arma::vec> origin;
	std::shared_ptr<arma::vec> direction;
	unsigned int row_index;
	unsigned int col_index;

	double range;
	Facet * hit_facet;

	double range_apriori;
	Facet * hit_facet_apriori;


	void find_intersect_with_facet(arma::vec & direction_in_target_frame,
	                               arma::vec & origin_in_target_frame,
	                               Facet * facet,
	                               bool computed_mes);
	bool intersection_inside(arma::vec & H, Facet * facet, double tol = 1e-7) ;


};




#endif