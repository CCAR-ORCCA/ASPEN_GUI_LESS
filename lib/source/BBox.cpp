#include "BBox.hpp"


BBox::BBox() {

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

void BBox::update(std::shared_ptr<Element> element) {

	arma::vec first_coordinates = element -> get_control_points() -> at(0) -> get_coordinates();

	this -> xmin = first_coordinates(0);
	this -> ymin = first_coordinates(1);
	this -> zmin = first_coordinates(2);

	this -> xmax = first_coordinates(0);
	this -> ymax = first_coordinates(1);
	this -> zmax = first_coordinates(2);

	for (unsigned int i = 1; i <  element -> get_control_points() -> size() ; ++i) {

		arma::vec coordinates = element -> get_control_points() -> at(i) -> get_coordinates();

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

void BBox::update(std::vector<std::shared_ptr<Element > > elements) {

	arma::vec first_coordinates = elements . at(0) -> get_control_points() -> at(0) -> get_coordinates();

	this -> xmin = first_coordinates(0);
	this -> ymin = first_coordinates(1);
	this -> zmin = first_coordinates(2);

	this -> xmax = first_coordinates(0);
	this -> ymax = first_coordinates(1);
	this -> zmax = first_coordinates(2);


	for (unsigned int j = 1; j < elements.size() ; ++j) {

		for (unsigned int i = 1; i <  elements[j] -> get_control_points() -> size() ; ++i) {

			arma::vec coordinates = elements[j] -> get_control_points() -> at(i) -> get_coordinates();

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
