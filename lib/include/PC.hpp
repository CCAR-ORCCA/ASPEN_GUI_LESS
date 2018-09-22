#ifndef HEADER_PC
#define HEADER_PC

#include <armadillo>
#include <memory>
#include <cassert>

#include "KDTreePC.hpp"
#include "KDTreeDescriptors.hpp"
#include "Ray.hpp"
#include "PointNormal.hpp"
#include "ShapeModelTri.hpp"

typedef typename std::pair<int, int > PointPair ;

class ShapeModelTri;
class Ray;

class PC {


public:

	/**
	Constructor
	@param focal_plane Pointer to focal plane whose individual rays impacting with the target
	yield the point cloud
	*/
	PC(std::vector<std::shared_ptr<Ray> > * focal_plane,int label_);

	PC(arma::mat & points,arma::mat & normals) ;

	/**
	/ Constructor
	/ @param los_dir Not used
	/ @param points 3-by-N matrix
	*/
	
	PC(arma::vec los_dir, arma::mat & points);

	PC(const std::vector<PointNormal> & point_normals);

	PC(std::vector< std::shared_ptr<PC> > & pcs,int points_retained = -1);


	/**
	Constructor
	@param filename path to OBJ file storing point coordinates
	*/
	PC(std::string filename);
	/*
	Constructor
	@param shape_model pointer to shape model
	*/
	PC(ShapeModelTri * shape_model);

	/**
	Computes point cloud surface normals using the provided los and neighborhood query radius
	@param los_dir direction of line of sight, from the instrument to the point cloud
	@param radius dimension of sphere within which the neighbors used in the normals computation live
	*/
	void construct_normals(arma::vec los_dir,double radius) ;

	/**
	Computes point cloud surface normals using the provided los and neighborhood query radius
	@param los_dir direction of line of sight, from the instrument to the point cloud
	@param N numbers of closest neighbors to each query point used in the normal vector computation
	*/
	void construct_normals(arma::vec los_dir,unsigned int N) ;



	// /**
	// Constructor
	// @param dcm DCM of the ICP rigid transform
	// @param x Translation term of the ICP rigid transform
	// @param destination_pc Pointer to destination PC
	// @param source_pc Pointer to destination PC
	// @param frame_graph Pointer to frame graph
	// */
	// PC(arma::mat & dcm,
	// 	arma::vec & x,
	// 	std::shared_ptr<PC> destination_pc,
	// 	std::shared_ptr<PC> source_pc,
	// 	FrameGraph * frame_graph);

	
	/**
	Returns the index to the PointNormal whose point is closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return Pointer to closest point
	*/
	int get_closest_point(const arma::vec & test_point) const;


	// /**
	// Appends N points from the provided point cloud to the present point cloud
	// @param other_pc Point cloud from which points must be included
	// @param N Number of points to be kept from the other point cloud
	// */
	// void append(std::shared_ptr<PC> other_pc, unsigned int N);


	/**
	Returns queried point
	@param index Index of the queried point
	@return queried point
	*/
	const PointNormal & get_point(unsigned int index) const;

	/**
	Returns queried point coordinates
	@param index Index of the queried point
	@return queried point coordinates
	*/
	arma::vec get_point_coordinates(unsigned int index) const;


	/**
	Returns queried point normal coordinates
	@param index Index of the queried point
	@return queried point normal coordinates
	*/
	arma::vec get_normal_coordinates(unsigned int index) const;

	/**
	Returns point cloud size
	@return point cloud size
	*/
	unsigned int size() const;


	/**
	Builds KD tree for closest-point queries
	*/
	void build_kd_tree_points();


	/**
	Return the coordinates of the point cloud's geometrical center
	@return coordinates of geometrical center
	*/
	arma::vec::fixed<3> get_center() const;

	/**
	Return the matrix directing the principal axes of the point cloud
	@return matrix of principal axes directing the point cloud
	*/
	arma::mat::fixed<3,3> get_principal_axes() const;

	// /**
	// Return the coordinates of the point cloud's bounding box geometrical center
	// @return coordinates of geometrical center
	// */
	// arma::vec get_bbox_center() const;


	// /**
	// Returns pointer the coordinates of the to queried point
	// @param index Index of the queried point
	// @return queried point coordinatess
	// */
	// arma::vec get_point_coordinates(unsigned int index) const;


	// /**
	// Returns normal of the queried point
	// @param index Index of the queried point
	// @return normal of queried point
	// */
	// arma::vec get_normal_coordinates(unsigned int index) const;

	/**
	Returns a map to the closest N PointNormal-s whose coordinates are closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return map to these closest points sorted by distance
	*/
	std::map<double,int > get_closest_N_points(const arma::vec & test_point, const unsigned int & N) const;



	/**
	Returns a map to the closest N PointNormal-s whose feature descriptors are closest to the provided test_point
	using the KD Tree search
	@param histogram N-by-1 histogram queried
	@return map to these closest points sorted by distance
	*/
	std::map<double,int > get_closest_N_features(const arma::vec & histogram,const int & N) const;

	// /**
	// Return size of point clouds
	// @param size of point cloud
	// */
	// unsigned int get_size() const;


	// /**
	// Returns length of diagonal
	// @return length of diagonal
	// */
	// double get_bbox_diagonal() const;

	// /**
	// Returns bbox dimensions
	// @return bbox dimensions along (x,y,z) axes
	// */
	// arma::vec get_bbox_dim() const;	

	void transform(const arma::mat & dcm, const arma::vec & x);

	/**
	Saves pc to file after applying a rigid transform
	@param path save path
	@param dcm DCM
	@param x Translation component
	@param save_normals true if normals should be added on the same row as
	the coordinates. cannot be true at the same time as format_like_obj!
	@param format_like_obj true if v tag should be added in front of 
	vertex coordinates. cannot be true at the same time as save_normals!
	*/
	void save(std::string path,
		arma::mat dcm = arma::eye<arma::mat>(3, 3),
		arma::vec x = arma::zeros<arma::vec>(3),
		bool save_normals = false,
		bool format_like_obj = true) const;

	/**
	Saves pc to file after applying a rigid transform
	@param path save path
	@param Points to save
	*/
	static void save(arma::mat & points,std::string path);


	std::string get_label() const;

	void save_point_descriptors(std::string path) const;

	/**
	Returns the points of $this that are within the sphere of specified radius centered at $test_point
	@param test_point query point
	@param radius non-negative search radius
	@return vector of indices of points in $this that satisfy || test_point - point_in_$this ||  < radius
	*/
	std::vector<int> get_nearest_neighbors_radius(const arma::vec & test_point, const double & radius) const;


	void compute_mean_feature_histogram();

	// static std::vector<PointPair>  find_pch_matches_kdtree(std::shared_ptr<PC> pc_source,std::shared_ptr<PC> pc_destination);


	static std::vector<PointPair>  find_pch_matches_kdtree(const std::shared_ptr<PC>&  pc_source,
		const std::shared_ptr<PC> & pc_destination);


	static void find_N_closest_pch_matches_kdtree(const std::shared_ptr<PC> & pc_source,
		const std::shared_ptr<PC> & pc_destination,const int N_closest_matches,std::vector< int > & active_source_points,
		std::map<int , std::vector< int> > & possible_matches);



	// static void save_pch_matches(const std::multimap<double,std::pair<int,int> > matches, std::string path);
	void compute_feature_descriptors(int type,bool keep_correlations,int N_bins,double neighborhood_radius,std::string pc_name);

	// void compute_neighborhoods(double radius);
	void compute_PFH(bool keep_correlations,int N_bins,double neighborhood_radius);
	void compute_FPFH(bool keep_correlations,int N_bins,double neighborhood_radius);
	void save_active_features(int index,std::string pc_name) const;

	// std::shared_ptr<KDTreePC> kdt_points_simple;
	
protected:

	void prune_features() ;


	// static std::vector<PointPair> generate_random_correspondance_table(const std::vector<std::shared_ptr< PointNormal > > & neighborhood,
	// 	const std::map<std::shared_ptr< PointNormal >, std::map<double,PointNormal * > > & pc0_to_pc1_potential_matches);
	// static double compute_neighborhood_consensus_ll(const std::vector<PointPair> & correspondance_table);
	
	std::vector<PointNormal> points_normals;

	std::shared_ptr<KDTreePC> kdt_points;
	std::shared_ptr<KDTreeDescriptors> kdt_descriptors;

	arma::vec mean_feature_histogram;

	std::string label;

	// double average_neighborhood_size;


};


#endif