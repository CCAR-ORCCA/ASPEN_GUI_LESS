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

typedef typename std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > PointPair ;

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
	Constructor
	@param los_dir Not used
	@param points 3-by-N matrix
	*/
	PC(arma::vec los_dir, arma::mat & points);

	PC(std::vector<std::shared_ptr<PointNormal> > point_normals);


	PC(std::vector< std::shared_ptr<PC> > & pcs,int points_retained = -1);



	PC(std::string filename);
	/*
	Constructor
	@param shape_model pointer to shape model
	*/
	PC(ShapeModelTri * shape_model);


	/**
	Constructor
	@param dcm DCM of the ICP rigid transform
	@param x Translation term of the ICP rigid transform
	@param destination_pc Pointer to destination PC
	@param source_pc Pointer to destination PC
	@param frame_graph Pointer to frame graph
	*/
	PC(arma::mat & dcm,
		arma::vec & x,
		std::shared_ptr<PC> destination_pc,
		std::shared_ptr<PC> source_pc,
		FrameGraph * frame_graph);

	
	/**
	Returns a pointer to the PointNormal whose point is closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return Pointer to closest point
	*/
	std::shared_ptr<PointNormal> get_closest_point(const arma::vec & test_point) const;


	/**
	Appends N points from the provided point cloud to the present point cloud
	@param other_pc Point cloud from which points must be included
	@param N Number of points to be kept from the other point cloud
	*/
	void append(std::shared_ptr<PC> other_pc, unsigned int N);


	/**
	Returns pointer to queried point
	@param index Index of the queried point
	@return queried point
	*/
	std::shared_ptr<PointNormal> get_point(unsigned int index) const;

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

	/**
	Return the coordinates of the point cloud's bounding box geometrical center
	@return coordinates of geometrical center
	*/
	arma::vec get_bbox_center() const;


	/**
	Returns pointer the coordinates of the to queried point
	@param index Index of the queried point
	@return queried point coordinatess
	*/
	arma::vec get_point_coordinates(unsigned int index) const;

	/**
	Returns normal of the queried point
	@param index Index of the queried point
	@return normal of queried point
	*/
	arma::vec get_point_normal(unsigned int index) const;


	/**
	Returns a pointer to the N PointNormal-s whose points are closest to the provided test_point
	using the KD Tree search
	@param test_point 3-by-1 vector queried
	@return Vector of pointers to closest points
	*/
	std::map<double,std::shared_ptr<PointNormal> > get_closest_N_points(const arma::vec & test_point, 
		const unsigned int & N) const;


	/**
	Return size of point clouds
	@param size of point cloud
	*/
	unsigned int get_size() const;


	/**
	Returns length of diagonal
	@return length of diagonal
	*/
	double get_bbox_diagonal() const;

	/**
	Returns bbox dimensions
	@return bbox dimensions along (x,y,z) axes
	*/
	arma::vec get_bbox_dim() const;	

	void transform(const arma::mat & dcm, const arma::vec & x);

	/**
	Returns the points in this point cloud
	@param points point in the point cloud
	*/
	std::vector< std::shared_ptr<PointNormal> > get_points() const;


	std::shared_ptr<PointNormal> get_best_match_feature_point(std::shared_ptr<PointNormal> other_point,double & distance) const;

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



	std::vector<std::shared_ptr<PointNormal> > get_points_in_sphere(
		arma::vec test_point, const double & radius) const;
	void compute_mean_feature_histogram();

	static std::vector<PointPair>  find_pch_matches(const PC & pc0,const PC & pc1);
	static std::vector<PointPair>  find_pch_matches(std::shared_ptr<PC> pc0,std::shared_ptr<PC> pc1);
	static std::vector<PointPair>  find_pch_matches_kdtree(std::shared_ptr<PC> pc0,std::shared_ptr<PC> pc1);

	static void save_pch_matches(const std::multimap<double,std::pair<int,int> > matches, std::string path);
	void compute_feature_descriptors(int type,bool keep_correlations,int N_bins,double neighborhood_radius);

	enum FeatureDescriptor { PFHDescriptor, FPFHDescriptor };

protected:

	void construct_kd_tree(std::vector< std::shared_ptr<PointNormal> > & points_normals);
	void construct_normals(arma::vec los);
	void prune_features() ;
	void save_active_features(int index) const;

	void compute_PFH(bool keep_correlations,
		int N_bins,double neighborhood_radius);


	void compute_FPFH(bool keep_correlations,
		int N_bins,double neighborhood_radius);

	std::shared_ptr<KDTreePC> kdt_points;
	std::shared_ptr<KDTreeDescriptors> kdt_descriptors;
	arma::vec mean_feature_histogram;

	arma::vec los;
	std::string label;

	double average_neighborhood_size;


};


#endif