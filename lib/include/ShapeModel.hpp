#ifndef HEADER_SHAPEMODEL
#define HEADER_SHAPEMODEL

#include <string>
#include <iostream>
#include <armadillo>
#include <set>
#include <map>
#include <limits>
#include "OMP_flags.hpp"
#include "FrameGraph.hpp"
#include "ControlPoint.hpp"
#include "KDTree_control_points.hpp"
#include "KDTree_shape.hpp"

class Ray ;
class Element;
class KDTree_shape;

/**
Declaration of the ShapeModel class. Base class for 
the implementation of shape model
*/
class ShapeModel {

public:

	/**
	Constructor
	*/
	ShapeModel();


	/**
	Constructor
	@param frame_graph Pointer to the graph storing
	reference frame relationships
	@param frame_graph Pointer to the reference frame graph
	*/
	ShapeModel(std::string ref_frame_name,
		FrameGraph * frame_graph);

	virtual void construct_kd_tree_shape() = 0;


	/**
	Returns the dimensions of the bounding box
	@param Bounding box dimension to be computed (xmin,ymin,zmin,xmax,ymax,zmax)
	*/
	void get_bounding_box(double * bounding_box,arma::mat M = arma::eye<arma::mat>(3,3)) const;


	/**
	Translates the shape model by x
	@param x translation vector applied to the coordinates of each control point
	*/
	void translate(arma::vec x);

	/**
	Rotates the shape model by 
	@param M rotation matrix
	*/
	void rotate(arma::mat M);

	
	/**
	Returns pointer to KDTree_shape member.
	@return pointer to KDtree_shape
	*/
	std::shared_ptr<KDTree_shape> get_KDTree_shape() const ;


	/**
	Returns the principal axes and principal moments of the shape model
	@param axes M as in X = MX' where X' is a position expressed in the principal frame
	@param moments dimensionless inertia moments in ascending order
	*/	
	void get_principal_inertias(arma::mat & axes,arma::vec & moments) const;



	/**
	Defines the reference frame attached to the shape model
	@param ref_frame Pointer to the reference frame attached
	to the shape model
	*/
	void set_ref_frame_name(std::string ref_frame_name);

	/**
	Returns the name of the reference frame attached to this
	ref frame
	@return name of reference frame
	*/
	std::string get_ref_frame_name() const;


	/**
	Returns pointer to prescribed control point
	@param i index of control point . must be between 0 and Nc - 1
	*/
	std::shared_ptr< ControlPoint>  get_control_point(unsigned int i) const ;


	/**
	Pointer to the shape model's control points
	@return vertices pointer to the control points
	*/
	std::vector<std::shared_ptr< ControlPoint> > * get_control_points();



	unsigned int get_control_point_index(std::shared_ptr<ControlPoint> point) const;

	/**
	Pointer to the shape model's element
	@return pointer to the elements
	*/
	std::vector<std::shared_ptr<Element> >  * get_elements();

	/**
	Augment the internal container storing elements with a new (and not already inserted)
	one
	@param facet pointer to the new element to be inserted
	*/
	void add_element(std::shared_ptr<Element> el);


	/**
	Returns the geometrical center of the shape
	@return geometrical center
	*/
	arma::vec get_center() const;
	
	

	/**
	Augment the internal container storing vertices with a new (and not already inserted)
	one
	@param control_point pointer to the new control point to be inserted
	*/
	void add_control_point(std::shared_ptr<ControlPoint> control_point);

	

	/**
	Returns number of elements
	@return number of elements
	*/
	unsigned int get_NElements() const  ;

	/**
	Returns number of control points
	@return number of control points
	*/
	unsigned int get_NControlPoints() const ;


	/**
	Constructs the KDTree holding the facets of the shape model for closest facet detection
	@param verbose true will save the bounding boxes to a file and display
	kd tree construction details
	*/
	void construct_kd_tree_control_points();

	/**
	Returns pointer to KDTree_control_points member.
	@return pointer to KDTree_control_points
	*/
	std::shared_ptr<KDTree_control_points> get_KDTree_control_points() const ;


	/**
	Computes the surface area of the shape model
	*/
	virtual void compute_surface_area() = 0;
	/**
	Computes the volume of the shape model
	*/
	virtual void compute_volume() = 0;
	/**
	Computes the center of mass of the shape model
	*/
	virtual void compute_center_of_mass() = 0;
	/**
	Computes the inertia tensor of the shape model
	*/
	virtual void compute_inertia() = 0;

	/**
	Finds the intersect between the provided ray and the shape model
	@param ray pointer to ray. If a hit is found, the ray's internal is changed to store the range to the hit point
	*/
	virtual bool ray_trace(Ray * ray) = 0;



	/**
	Constructs a connectivity table associated a control point pointer to its index
	in this shape model control points vector
	*/
	void initialize_index_table();

	/**
	Returns the non-dimensional inertia tensor of the body in the body-fixed
	principal axes. (rho == 1, l = (volume)^(1/3))
	@return principal inertia tensor
	*/
	arma::mat get_inertia() const;


protected:

	std::vector<std::shared_ptr<Element  > > elements;
	std::vector<std::shared_ptr< ControlPoint> >  control_points;
	std::shared_ptr<KDTree_control_points> kdt_control_points = nullptr;
	std::shared_ptr<KDTree_shape> kdt_facet = nullptr;

	std::map<std::shared_ptr<ControlPoint> ,unsigned int> pointer_to_global_index;

	FrameGraph * frame_graph;
	std::string ref_frame_name;

	arma::mat inertia;


};

#endif