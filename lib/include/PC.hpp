#ifndef HEADER_PC
#define HEADER_PC

#include <armadillo>
#include <memory>
#include <cassert>

#include "KDTreePC.hpp"
#include "Ray.hpp"
#include "PointNormal.hpp"
#include "ShapeModelTri.hpp"

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
	arma::vec get_center() const;


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
	std::vector<std::shared_ptr<PointNormal> > get_closest_N_points(arma::vec test_point, unsigned int N) const;


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



protected:

	void construct_kd_tree(std::vector< std::shared_ptr<PointNormal> > & points_normals);
	void construct_normals(arma::vec los);

	std::shared_ptr<KDTreePC> kdt_points;

	arma::vec los;
	std::string label;


};


#endif