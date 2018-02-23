
#ifndef HEADER_SHAPEMODELIMPORTER
#define HEADER_SHAPEMODELIMPORTER


#include <armadillo>
#include "omp.h"
#include <boost/progress.hpp>

class ShapeModelTri;
class ShapeModelBezier;



class ShapeModelImporter {

public:

	/**
	Constructor
	@param filename absolute or relative path to the OBJ file to be read
	@param scaling_factor 1 if provided file uses meters, 1000 if km are used, ...
	@param as_is true if the provided shape model should not barycentered or aligned with principal axes
	*/
	ShapeModelImporter(std::string filename, double scaling_factor, bool as_is);

	/**
	Reads-in an OBJ file storing the polyhedral shape model info and sets the field of
	$shape_model to the corresponding values
	@param shape_model Pointer to the shape model to receive the read data
	*/
	void load_obj_shape_model(ShapeModelTri * shape_model) const;

	/**
	Reads-in an .b file storing the bezier shape model info and sets the field of
	$shape_model to the corresponding values
	@param shape_model Pointer to the shape model to receive the read data
	*/
	void load_bezier_shape_model(ShapeModelBezier * shape_model) const;

protected:
	std::string filename;
	double scaling_factor;
	bool as_is;

};

#endif