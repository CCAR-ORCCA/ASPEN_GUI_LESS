#include "ICP.hpp"

ICP::ICP(arma::vec * source_points, arma::vec * destination_points) {
	this -> source_points = source_points;
	this -> destination_points = destination_points;

	// It would probably be helpful to have a kd-tree like structure to quickly find closest points

	// The normals are computed
	this -> compute_normals();

}

void ICP::compute_normals(){

}