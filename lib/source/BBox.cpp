#include "BBox.hpp"
#include <armadillo>
#include "Element.hpp"
#include "ShapeModel.hpp"


BBox::BBox(ShapeModel * owning_shape) {
	this -> owning_shape = owning_shape;
}


void BBox::reset_bbox(){
	this -> xmin = this -> ymin = this -> zmin = std::numeric_limits<double>::infinity();
	this -> xmax = this -> ymax = this -> zmax = - std::numeric_limits<double>::infinity();
}


void BBox::print() const {

	std::cout << std::endl << "Bounding box coordinates:" << std::endl;
	std::cout << this -> xmin << " " << this -> xmax << std::endl;
	std::cout << this -> ymin << " " << this -> ymax << std::endl;
	std::cout << this -> zmin << " " << this -> zmax << std::endl  << std::endl;

	std::cout << "Bounding box volume:" << std::endl;

	double x_axis_length = this -> xmax - this -> xmin;
	double y_axis_length = this -> ymax - this -> ymin;
	double z_axis_length = this -> zmax - this -> zmin;

	std::cout << x_axis_length * y_axis_length * z_axis_length << std::endl  << std::endl;

}

void BBox::update(int element_index) {

	const std::vector<int> & control_points = this -> owning_shape -> get_element_control_points(element_index);

	const arma::vec::fixed<3> & first_coordinates = this -> owning_shape -> get_control_point_coordinates(control_points[0]);
	
	this -> xmin = std::min(first_coordinates(0),this -> xmin);
	this -> ymin = std::min(first_coordinates(1),this -> ymin);
	this -> zmin = std::min(first_coordinates(2),this -> zmin);

	this -> xmax = std::max(first_coordinates(0),this -> xmax);
	this -> ymax = std::max(first_coordinates(1),this -> ymax);
	this -> zmax = std::max(first_coordinates(2),this -> zmax);

	for (unsigned int i = 1; i <  control_points.size() ; ++i) {

		const arma::vec::fixed<3> & coordinates = this -> owning_shape -> get_control_point_coordinates(control_points[i]);

		if (coordinates(0) > this -> xmax) {
			this -> xmax = coordinates(0);
		}
		if (coordinates(0) < this -> xmin) {
			this -> xmin = coordinates(0);
		}

		if (coordinates(1) > this -> ymax) {
			this -> ymax = coordinates(1);
		}
		if (coordinates(1) < this -> ymin) {
			this -> ymin = coordinates(1);
		}

		if (coordinates(2) > this -> zmax) {
			this -> zmax = coordinates(2);
		}

		if (coordinates(2) < this -> zmin) {
			this -> zmin = coordinates(2);
		}

	}


}

void BBox::update(std::vector<int> & element_indices) {


	const std::vector<int> & first_element_control_points = this -> owning_shape -> get_element_control_points(element_indices[0]);


	const arma::vec::fixed<3> & first_coordinates = this -> owning_shape -> get_control_point_coordinates(first_element_control_points[0]);

	this -> xmin = first_coordinates(0);
	this -> ymin = first_coordinates(1);
	this -> zmin = first_coordinates(2);

	this -> xmax = first_coordinates(0);
	this -> ymax = first_coordinates(1);
	this -> zmax = first_coordinates(2);

	for (unsigned int j = 1; j < element_indices.size() ; ++j) {
		this -> update(element_indices[j]);
	}


}

unsigned int BBox::get_longest_axis() const {

	double x_axis_length = this -> xmax - this -> xmin;
	double y_axis_length = this -> ymax - this -> ymin;
	double z_axis_length = this -> zmax - this -> zmin;

	if (x_axis_length >= std::max(y_axis_length, z_axis_length)) {
		return 0;
	}
	else if (y_axis_length >= std::max(x_axis_length, z_axis_length)) {
		return 1;
	}

	else {
		return 2;
	}

}

double BBox::get_xmin() const {
	return this -> xmin;
}
double BBox::get_xmax() const {
	return this -> xmax;
}
double BBox::get_ymin() const {
	return this -> ymin;
}
double BBox::get_ymax() const {
	return this -> ymax;
}
double BBox::get_zmin() const {
	return this -> zmin;
}
double BBox::get_zmax() const {
	return this -> zmax;
}



void BBox::save_to_file(std::string path) const {

	arma::mat r = {	{xmin, ymin, zmin},
	{xmax, ymin, zmin},
	{xmax, ymax, zmin},
	{xmax, ymax, zmax},
	{xmin, ymax, zmax},
	{xmin, ymin, zmax},
	{xmax, ymin, zmax},
	{xmin, ymax, zmin}
};


std::ofstream shape_file;
shape_file.open(path);

for (unsigned int vertex_index = 0;
	vertex_index < r.n_rows;
	++vertex_index) {
	shape_file << "v " << r.row(vertex_index)(0) << " " <<  r.row(vertex_index)(1) << " " <<  r.row(vertex_index)(2) << std::endl;
}







}
