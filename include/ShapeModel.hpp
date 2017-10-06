
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

	std::vector<std::shared_ptr<Element> > * get_elements();


	/**
	Returns number of elements
	@return number of elements
	*/
	virtual unsigned int get_NElements() const = 0 ;

	/**
	Returns number of control points
	@return number of control points
	*/
	virtual unsigned int get_NControlPoints() const = 0;

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
	@param computed_mes true if the target is the estimated shape
	*/
	virtual bool ray_trace(Ray * ray,bool computed_mes) = 0;


protected:


	FrameGraph * frame_graph;
	std::string ref_frame_name;

};

#endif