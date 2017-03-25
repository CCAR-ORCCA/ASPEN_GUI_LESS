#include "Lidar.hpp"

Lidar::Lidar(
    FrameGraph * frame_graph,
    std::string ref_frame_name,
    double fov_y,
    double fov_z,
    unsigned int res_y,
    unsigned int res_z,
    double f) {

	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
	this -> f = f;
	this -> fov_y = fov_y;
	this -> fov_z = fov_z;
	this -> res_y = res_y;
	this -> res_z = res_z;



	for (unsigned int y_index = 0; y_index < this -> res_y; ++y_index) {

		std::vector<std::shared_ptr<Ray> > h_row;

		for (unsigned int z_index = 0; z_index < this -> res_z; ++z_index) {
			h_row.push_back(std::make_shared<Ray>(Ray(y_index, z_index, this)));
		}

		this -> focal_plane.push_back(h_row);
	}
}

double Lidar::get_res_y() const {
	return (double)(this -> res_y);
}

double Lidar::get_res_z() const {
	return (double)(this -> res_z);
}

double Lidar::get_focal_length() const {
	return this -> f;
}

FrameGraph * Lidar::get_frame_graph() {
	return this -> frame_graph;
}


double Lidar::get_fov_z(bool use_rad ) const {
	if (use_rad == true) {
		return arma::datum::pi / 180 * this -> fov_z;
	}
	else {
		return this -> fov_z;
	}
}

double Lidar::get_fov_y(bool use_rad ) const {
	if (use_rad == true) {
		return arma::datum::pi / 180 * this -> fov_y;
	}
	else {
		return this -> fov_y;
	}
}


double Lidar::get_size_z() const {

	return 2 * this -> get_focal_length() * std::tan(this ->  get_fov_z() / 2);

}

double Lidar::get_size_y() const {
	return 2 * this -> get_focal_length() * std::tan(this ->  get_fov_y() / 2);
}

void Lidar::send_flash(ShapeModel * shape_model) {
	this -> shape_model = shape_model;

	unsigned int y_res = this -> res_y;
	unsigned int z_res = this -> res_z;

	// #pragma omp parallel for if(USE_OMP_LIDAR)
	for (unsigned int y_index = 0; y_index < y_res; ++y_index) {

		// Segfaults ??
		for (unsigned int z_index = 0; z_index < z_res; ++z_index) {
			this -> focal_plane[y_index][z_index] -> brute_force_ray_casting();
		}
	}

}


ShapeModel * Lidar::get_shape_model() {
	return this -> shape_model;
}

std::string Lidar::get_ref_frame_name() const {
	return this -> ref_frame_name;
}

void Lidar::save_focal_plane_range(std::string path) const {

	// std::ofstream pixel_location_file;
	// pixel_location_file.open(path);
	Gnuplot gp;

	std::vector<boost::tuple<unsigned int, unsigned int, double> > range_tuple_vec;


	for (unsigned int y_index = 0; y_index < this -> res_y; ++y_index) {

		for (unsigned int z_index = 0; z_index < this -> res_z; ++z_index) {
			// Cheap and dirty way to get rid of the inf
			if (this -> focal_plane[y_index][z_index] -> get_range() < 1e10)
				// pixel_location_file << y_index << " " << z_index << " " << this -> focal_plane[y_index][z_index] -> get_range() << "\n";
				range_tuple_vec.push_back(boost::make_tuple(
				                              y_index,
				                              z_index,
				                              this -> focal_plane[y_index][z_index] -> get_range()
				                          ));

		}

	}
	// pixel_location_file.close();
	gp << "set view map\n";
	gp << "set dgrid3d\n";
	gp << "set xrange [-1:" << std::to_string(this -> res_y + 1) << "]\nset yrange [-2:" << std::to_string(this -> res_z + 1) << "]\n";

	gp << "splot" << gp.file1d(range_tuple_vec) << "using 1:2:3 with pm3d\n";




}



void Lidar::save_pixel_location(std::string path) const {

	std::ofstream pixel_location_file;
	pixel_location_file.open(path);

	// The focal plane is populated row by row
	for (unsigned int y_index = 0; y_index < this -> res_y; ++y_index) {


		// column by column
		for (unsigned int z_index = 0; z_index < this -> res_z; ++z_index) {

			pixel_location_file << this -> focal_plane[y_index][z_index] -> get_origin() -> colptr(0)[0] << " "
			                    << this -> focal_plane[y_index][z_index] -> get_origin() -> colptr(0)[1] << " "
			                    << this -> focal_plane[y_index][z_index] -> get_origin() -> colptr(0)[2] << "\n";

		}



	}
	pixel_location_file.close();

}
