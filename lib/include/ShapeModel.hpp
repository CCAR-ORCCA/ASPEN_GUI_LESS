
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


class Ray ;



class Element;

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
	Pointer to the shape model's control points
	@return vertices pointer to the control points
	*/
	std::vector<std::shared_ptr< ControlPoint> > * get_control_points();

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
	void construct_kd_tree_control_points(bool verbose = false);

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


protected:

	std::vector<std::shared_ptr<Element  > > elements;
	std::vector<std::shared_ptr< ControlPoint> >  control_points;
	std::shared_ptr<KDTree_control_points> kdt_control_points = nullptr;


	FrameGraph * frame_graph;
	std::string ref_frame_name;

};

#endif