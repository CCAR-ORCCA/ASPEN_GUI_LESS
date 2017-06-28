#ifndef HEADER_PC
#define HEADER_PC

#include <armadillo>
#include <memory>

#include "KDTree_pc.hpp"
#include "Ray.hpp"
#include "PointNormal.hpp"


class PC {

public:

	/**
	Constructor
	@param los_dir Not used
	@param focal_plane Pointer to focal plane whose individual rays impacting with the target
	yield the point cloud
	*/
	PC(arma::vec los_dir, std::vector<std::vector<std::shared_ptr<Ray> > > * focal_plane);
	
	/**
	Constructor
	@param los_dir Not used
	@param points 3-by-N matrix
	*/
	PC(arma::vec los_dir, arma::mat & points);

	/**
	Returns a pointer to the PointNormal whose point is closest to the provided test_point
	using a brute force approach
	@param test_point 3-by-1 vector queried
	@return Pointer to closest point
	*/
	std::shared_ptr<PointNormal> get_closest_point_index_brute_force(arma::vec & test_point) const;

	/**
	Returns a pointer to the N PointNormal-s whose points are closest to the provided test_point
	using a brute force approach
	@param test_point 3-by-1 vector queried
	@return Vector of pointers to closest points
	*/
	std::vector<std::shared_ptr<PointNormal> > get_closest_N_points_brute_force(arma::vec & test_point, unsigned int N) const ;

	/**
	Returns a pointer to the PointNormal whose point is closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return Pointer to closest point
	*/
	std::shared_ptr<PointNormal> get_closest_point(arma::vec & test_point) const;

	/**
	Returns a pointer to the N PointNormal-s whose points are closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return Vector of pointers to closest points
	*/
	std::vector<std::shared_ptr<PointNormal> > get_closest_N_points(arma::vec & test_point, unsigned int N) const;


protected:

	void construct_kd_tree(std::vector< std::shared_ptr<PointNormal> > & points_normals);
	void construct_normals(arma::vec & los);

	std::shared_ptr<KDTree_pc> kd_tree;

};


#endif