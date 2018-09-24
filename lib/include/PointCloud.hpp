#ifndef HEADER_POINT_CLOUD
#define HEADER_POINT_CLOUD

#include <armadillo>
#include <memory>
#include <cassert>

#include "KDTree.hpp"


typedef typename std::pair<int, int > PointPair ;

template <class T> class PointCloud {

public:
	PointCloud(int size);
	PointCloud(const std::vector<T> & points);
	PointCloud(std::vector< std::shared_ptr< PointCloud < T> > > & pcs,int points_retained);

	PointCloud(std::string filename);

	/**
	Returns the index to point cloud element whose point is closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return Pointer to closest point
	*/
	int get_closest_point(const arma::vec & test_point) const;

	/**
	Returns queried point
	@param index Index of the queried point
	@return queried point
	*/
	const T & get_point(unsigned int index) const;

	/**
	Returns queried point
	@param index Index of the queried point
	@return queried point
	*/
	T & get_point(unsigned int index) ;


	/**
	Returns point cloud size
	@return point cloud size
	*/
	unsigned int size() const;

	/**
	Returns a map to the closest N T-s whose coordinates are closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return map to these closest points sorted by distance
	*/
	std::map<double,int > get_closest_N_points(const arma::vec & test_point, const unsigned int & N) const;

	/**
	Get label identifying the point cloud
	*/
	std::string get_label() const;

	/**
	Returns a constant reference to the coordinates of the queried point at the provided index
	@param index
	@return vector of coordinates
	*/
	const arma::vec & get_point_coordinates(int i) const;

	/**
	Returns the points of $this that are within the sphere of specified radius centered at $test_point
	@param test_point query point
	@param radius non-negative search radius
	@return vector of indices of points in $this that satisfy || test_point - point_in_$this ||  < radius
	*/
	std::vector<int> get_nearest_neighbors_radius(const arma::vec & test_point, const double & radius) const;


	/**
	Adds an element to the point cloud
	*/
	void push_back(const T & point);

	/**
	Builds KD Tree
	*/
	void build_kdtree();

	/**
	Subscript operator accessing the underlying point vector
	*/
	T & operator[] (const int index);




protected:

	std::vector<T> points;
	std::shared_ptr< KDTree< PointCloud<T> > > kdt;
	arma::vec mean_feature_histogram;
	std::string label;


};


#endif