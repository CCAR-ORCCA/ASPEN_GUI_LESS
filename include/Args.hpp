#ifndef HEADER_ARGS
#define HEADER_ARGS

#include "FrameGraph.hpp"
#include "ShapeModel.hpp"
#include "Interpolator.hpp"

class Args {

public:

	Args(double density,
	     FrameGraph * frame_graph,
	     ShapeModel * shape_model) {
		this -> density = density;
		this -> frame_graph = frame_graph;
		this -> shape_model = shape_model;
		this -> stopping_bool = false;

	}

	double get_density() const {
		return this -> density;
	}

	FrameGraph * get_frame_graph() {
		return this -> frame_graph;
	}

	ShapeModel * get_shape_model() {
		return this -> shape_model;
	}

	void set_interpolator(Interpolator * interpolator) {
		this -> interpolator = interpolator;
	}

	void set_stopping_bool(bool stop) {
		this -> stopping_bool = stop;
	}

	bool get_stopping_bool() {
		return this -> stopping_bool;
	}


	Interpolator * get_interpolator() {
		return this -> interpolator;
	}

protected:
	double density;
	bool stopping_bool;
	FrameGraph * frame_graph;
	ShapeModel * shape_model;
	Interpolator * interpolator;


};

#endif
