
#ifndef HEADER_SHAPEMODELIMPORTER
#define HEADER_SHAPEMODELIMPORTER

#include "ShapeModel.hpp"
#include <armadillo>
#include "omp.h"
#include <boost/progress.hpp>


class ShapeModelImporter{

public:

	/**
	Constructor
	@param filename absolute or relative path to the OBJ file to be read
	*/
	ShapeModelImporter(std::string filename);

	/**
	Reads-in an OBJ file storing the shape model info and sets the field of 
	$shape_model to the corresponding values
	@param shape_model Pointer to the shape model to receive the read data
	*/
	void load_shape_model(ShapeModel * shape_model) const;

protected:
	std::string filename;

};

#endif