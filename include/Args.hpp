#ifndef HEADER_ARGS
#define HEADER_ARGS

#include "FrameGraph.hpp"
#include "ShapeModel.hpp"
#include "Interpolator.hpp"

class Args {

public:


	void set_frame_graph(FrameGraph * frame_graph) {
		this -> frame_graph = frame_graph;
	}

	void set_shape_model(ShapeModel * shape_model) {
		this -> shape_model = shape_model;
	}

	void set_density(double density) {
		this -> density = density;
	}

	void set_e(double e) {
		this -> eccentricity = e;
	}

	void set_mu(double mu) {
		this -> mu = mu;
	}

	void set_sma(double sma) {
		this -> sma = sma;
	}

	double get_sma() const {
		return this -> sma;
	}


	double get_mu() const {
		return this -> mu;
	}

	double get_density() const {
		return this -> density;
	}


	double get_e() const {
		return this -> eccentricity;
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

	bool get_is_attitude_bool() const {
		return this -> is_attitude_bool;
	}

	void set_is_attitude_bool(bool is_attitude) {
		this -> is_attitude_bool = is_attitude;
	}

	Interpolator * get_interpolator() {
		return this -> interpolator;
	}

	Interpolator * get_interp_s() const {
		return this -> interp_s;
	}

	void set_interp_s(Interpolator * interp_s) {
		this -> interp_s = interp_s;
	}

	void set_time(double time) {
		this -> time = time;
	}

	double get_time() const {
		return this -> time;
	}

	void set_minimum_elevation(double el) {
		this -> minimum_elevation = el;
	}
	
	double get_minimum_elevation() const {
		return this -> minimum_elevation;
	}

protected:
	double density;
	double eccentricity;
	double mu;
	double sma;
	double time;
	double minimum_elevation;
	bool stopping_bool = false;
	FrameGraph * frame_graph;
	ShapeModel * shape_model;
	Interpolator * interpolator;
	Interpolator * interp_s;

	bool is_attitude_bool = false;
	arma::mat * s;


};

#endif
