#include "Lidar.hpp"

Lidar::Lidar(
    FrameGraph * frame_graph,
    std::string ref_frame_name,
    double fov_y,
    double fov_z,
    unsigned int res_y,
    unsigned int res_z,
    double f,
    double freq) {

	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
	this -> f = f;
	this -> fov_y = fov_y;
	this -> fov_z = fov_z;
	this -> res_y = res_y;
	this -> res_z = res_z;
	this -> freq = freq;



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

double Lidar::get_frequency() const {
	return this -> freq;
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

std::pair<double, double> Lidar::save_focal_plane_range(std::string path) const {

	std::ofstream pixel_location_file;
	pixel_location_file.open(path + ".txt");

	double r_max = 	- std::numeric_limits<float>::infinity();
	double r_min =  std::numeric_limits<float>::infinity();


	for (unsigned int y_index = 0; y_index < this -> res_y; ++y_index) {


		pixel_location_file << this -> focal_plane[y_index][0] -> get_range() ;


		for (unsigned int z_index = 1; z_index < this -> res_z; ++z_index) {

			double range = this -> focal_plane[y_index][z_index] -> get_range() ;

			pixel_location_file << " " << range ;
			if (range > r_max)
				r_max = range;
			if (range < r_min)
				r_min = range;

		}

		pixel_location_file << "\n";


	}

	pixel_location_file.close();
	return std::make_pair(r_min, r_max);

}

void Lidar::plot_ranges(std::string path) const {

	std::pair<double, double> range_lims = this -> save_focal_plane_range(path);
	std::vector<std::string> script;
	script.push_back("set terminal png");
	script.push_back("set output '" + path + ".png'");

	script.push_back("set title ''");
	script.push_back("set view map");
	script.push_back("set palette rgb 33, 13, 10");
	script.push_back("set palette negative");
	script.push_back("set xrange [" + std::to_string(-1) + ":" + std::to_string(this -> res_z) + "]");
	script.push_back("set yrange [" + std::to_string(-1) + ":" + std::to_string(this -> res_y) + "]");
	script.push_back("set cbrange [" + std::to_string(range_lims.first) + ":" + std::to_string(range_lims.second) + "]");
	script.push_back("set size square");

	script.push_back("plot '" + path + ".txt' matrix with image notitle");


	GNUPlot plotter;
	plotter.open();
	plotter.execute(script);
	plotter.write("exit");
	plotter.flush();
	plotter.close();



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
