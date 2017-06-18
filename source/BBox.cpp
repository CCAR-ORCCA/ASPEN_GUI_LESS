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

void BBox::update(Facet * facet) {

	arma::vec * first_coordinates = facet -> get_vertices() -> at(0) -> get_coordinates();

	this -> xmin = first_coordinates -> at(0);
	this -> ymin = first_coordinates -> at(1);
	this -> zmin = first_coordinates -> at(2);

	this -> xmax = first_coordinates -> at(0);
	this -> ymax = first_coordinates -> at(1);
	this -> zmax = first_coordinates -> at(2);

	for (unsigned int i = 1; i <  facet -> get_vertices() -> size() ; ++i) {

		arma::vec * coordinates = facet -> get_vertices() -> at(i) -> get_coordinates();

		if (coordinates -> at(0) > this -> xmax) {
			this -> xmax = coordinates -> at(0);
		}
		if (coordinates -> at(0) < this -> xmin) {
			this -> xmin = coordinates -> at(0);
		}

		if (coordinates -> at(1) > this -> ymax) {
			this -> ymax = coordinates -> at(1);
		}
		if (coordinates -> at(1) < this -> ymin) {
			this -> ymin = coordinates -> at(1);
		}

		if (coordinates -> at(2) > this -> zmax) {
			this -> zmax = coordinates -> at(2);
		}

		if (coordinates -> at(2) < this -> zmin) {
			this -> zmin = coordinates -> at(2);
		}

	}

	// this -> make_consistent();

}

void BBox::update(std::vector<Facet * > facets) {

	arma::vec * first_coordinates = facets . at(0) -> get_vertices() -> at(0) -> get_coordinates();

	this -> xmin = first_coordinates -> at(0);
	this -> ymin = first_coordinates -> at(1);
	this -> zmin = first_coordinates -> at(2);

	this -> xmax = first_coordinates -> at(0);
	this -> ymax = first_coordinates -> at(1);
	this -> zmax = first_coordinates -> at(2);


	for (unsigned int j = 1; j < facets.size() ; ++j) {

		for (unsigned int i = 1; i <  facets[j] -> get_vertices() -> size() ; ++i) {

			arma::vec * coordinates = facets[j] -> get_vertices() -> at(i) -> get_coordinates();

			if (coordinates -> at(0) > this -> xmax) {
				this -> xmax = coordinates -> at(0);
			}

			if (coordinates -> at(0) < this -> xmin) {
				this -> xmin = coordinates -> at(0);
			}

			if (coordinates -> at(1) > this -> ymax) {
				this -> ymax = coordinates -> at(1);
			}

			if (coordinates -> at(1) < this -> ymin) {
				this -> ymin = coordinates -> at(1);
			}

			if (coordinates -> at(2) > this -> zmax) {
				this -> zmax = coordinates -> at(2);
			}

			if (coordinates -> at(2) < this -> zmin) {
				this -> zmin = coordinates -> at(2);
			}

		}
	}

	// this -> make_consistent();

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


void BBox::make_consistent() {

	double x_axis_length = this -> xmax - this -> xmin;
	double y_axis_length = this -> ymax - this -> ymin;
	double z_axis_length = this -> zmax - this -> zmin;

	arma::vec axes_length = {x_axis_length, y_axis_length, z_axis_length};
	arma::uvec axes_length_sorted_indices = arma::sort_index(axes_length);

	// Increase of the other two axes based on the length
	// of the longest one
	double shortest_axis_increase = 0.1 * axes_length(axes_length_sorted_indices(2));

	switch (axes_length_sorted_indices(0)) {
	case 0:
		this -> xmax += shortest_axis_increase / 2;
		this -> xmin -= shortest_axis_increase / 2;

		break;

	case 1:
		this -> ymax += shortest_axis_increase / 2;
		this -> ymin -= shortest_axis_increase / 2;
		break;


	case 2:
		this -> zmax += shortest_axis_increase / 2;
		this -> zmin -= shortest_axis_increase / 2;
		break;




	}

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
