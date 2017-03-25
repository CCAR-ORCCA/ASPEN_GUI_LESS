#include "Filter.hpp"

Filter::Filter(Lidar * lidar,
               ShapeModel * true_shape_model,
               ShapeModel * estimated_shape_model,
               double t0,
               double tf,
               double dt) {
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> t0 = t0;
	this -> tf = tf;
	this -> dt = dt;
}

