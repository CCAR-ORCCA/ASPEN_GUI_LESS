#include "Lidar.hpp"

Lidar::Lidar(
    FrameGraph * frame_graph,
    std::string ref_frame_name,
    double fov_y,
    double fov_z,
    unsigned int row_count,
    unsigned int col_count,
    double f,
    double freq) {

	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
	this -> f = f;
	this -> fov_y = fov_y;
	this -> fov_z = fov_z;
	this -> row_count = row_count;
	this -> col_count = col_count;
	this -> freq = freq;




	for (unsigned int y_index = 0; y_index < this -> row_count; ++y_index) {

		std::vector<std::shared_ptr<Ray> > h_row;

		for (unsigned int z_index = 0; z_index < this -> col_count; ++z_index) {
			h_row.push_back(std::make_shared<Ray>(Ray(y_index, z_index, this)));
		}

		this -> focal_plane.push_back(h_row);
	}
}

double Lidar::get_row_count() const {
	return (double)(this -> row_count);
}

double Lidar::get_col_count() const {
	return (double)(this -> col_count);
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

void Lidar::send_flash(ShapeModel * shape_model, bool computed_mes, bool store_mes) {

	this -> shape_model = shape_model;

	unsigned int y_res = this -> row_count;
	unsigned int z_res = this -> col_count;

	bool hit;

	for (unsigned int y_index = 0; y_index < y_res; ++y_index) {

		for (unsigned int z_index = 0; z_index < z_res; ++z_index) {

			// Range measurements is reset
			this -> focal_plane[y_index][z_index] -> reset(computed_mes);

			if (shape_model -> has_kd_tree()) {
				hit = shape_model -> get_kdtree().get() -> hit(shape_model -> get_kdtree().get(),
				        this -> focal_plane[y_index][z_index].get());
			}

			else {
				hit = this -> focal_plane[y_index][z_index] -> brute_force_ray_casting(computed_mes);

			}



			// If true, all the measurements are stored
			if (store_mes == true) {

				if (hit) {

					this -> surface_measurements.push_back(
					    this -> focal_plane[y_index][z_index] -> get_true_range() *
					    (*this -> focal_plane[y_index][z_index] -> get_direction_target_frame())
					    +  *this -> focal_plane[y_index][z_index] -> get_origin_target_frame());

				}
			}
		}
	}

	arma::vec u = {1, 0, 0};
	if (this -> destination_pc == nullptr) {
		this -> destination_pc = std::make_shared<PC>(PC(u, &this -> focal_plane, this -> frame_graph));
	}
	else {
		if (this -> source_pc != nullptr) {
			this -> destination_pc = this -> source_pc;
		}
		this -> source_pc = std::make_shared<PC>(PC(u, &this -> focal_plane, this -> frame_graph));
	}
}


ShapeModel * Lidar::get_shape_model() {
	return this -> shape_model;
}

std::string Lidar::get_ref_frame_name() const {
	return this -> ref_frame_name;
}




std::pair<double, double> Lidar::save_true_range(std::string path) const {

	std::ofstream pixel_location_file;
	pixel_location_file.open(path + ".txt");

	double r_max = 	- std::numeric_limits<float>::infinity();
	double r_min =  std::numeric_limits<float>::infinity();


	for (unsigned int y_index = 0; y_index < this -> row_count; ++y_index) {

		if (std::abs(this -> focal_plane[y_index][0] -> get_true_range()) < 1e10 )
			pixel_location_file << this -> focal_plane[y_index][0] -> get_true_range();
		else
			pixel_location_file << "nan";


		for (unsigned int z_index = 1; z_index < this -> col_count; ++z_index) {

			double range = this -> focal_plane[y_index][z_index] -> get_true_range() ;
			if (std::abs(range) < 1e10 ) {

				pixel_location_file << " " << range ;
				if (range > r_max) {
					r_max = range;
				}
				if (range < r_min ) {
					r_min = range;
				}
			}
			else
				pixel_location_file << " nan";

		}

		pixel_location_file << "\n";


	}

	pixel_location_file.close();

	return std::make_pair(r_min, r_max);

}


std::pair<double, double> Lidar::save_computed_range(std::string path) const {

	std::ofstream pixel_location_file;
	pixel_location_file.open(path + ".txt");

	double r_max = 	- std::numeric_limits<float>::infinity();
	double r_min =  std::numeric_limits<float>::infinity();


	for (unsigned int y_index = 0; y_index < this -> row_count; ++y_index) {

		if (std::abs(this -> focal_plane[y_index][0] -> get_computed_range()) < 1e10 )
			pixel_location_file << this -> focal_plane[y_index][0] -> get_computed_range();
		else
			pixel_location_file << "nan";



		for (unsigned int z_index = 1; z_index < this -> col_count; ++z_index) {

			double range = this -> focal_plane[y_index][z_index] -> get_computed_range() ;


			if (std::abs(range) < 1e10 ) {

				pixel_location_file << " " << range ;
				if (range > r_max)
					r_max = range;
				if (range < r_min )
					r_min = range;
			}
			else
				pixel_location_file << " nan";

		}

		pixel_location_file << "\n";


	}

	pixel_location_file.close();

	return std::make_pair(r_min, r_max);

}

Ray * Lidar::get_ray(unsigned int row_index, unsigned int col_index) {
	return this -> focal_plane[row_index][col_index].get();
}

void Lidar::save_range_residuals_per_facet(std::string path, std::map<Facet * , std::vector<double> > & facets_to_residuals) const {


	std::ofstream facets_residuals_file;
	facets_residuals_file.open(path + ".txt");

	unsigned int facet_index = 0;
	for (auto const & facet_pair : facets_to_residuals) {

		for (unsigned int res_index = 0; res_index < facet_pair.second.size(); ++ res_index) {
			facets_residuals_file << facet_index << " " << facet_pair.second[res_index] << "\n";

		}

		++facet_index;
	}

	facets_residuals_file.close();

}

void Lidar::plot_range_residuals_per_facet(std::string path) {



	std::vector<std::string> script;
	script.push_back("set terminal png");
	script.push_back("set output '" + path + ".png'");

	script.push_back("set title ''");
	script.push_back("set view map");

	script.push_back("stats '" + path + ".txt' using 1");
	script.push_back("set xrange [STATS_min - 1:STATS_max + 1]");
	script.push_back("plot '" + path + ".txt' with points notitle");

	script.push_back("replot");

	GNUPlot plotter;
	plotter.open();
	plotter.execute(script);
	plotter.write("exit");
	plotter.flush();
	plotter.close();


}

std::pair<double, double> Lidar::save_range_residuals(std::string path) const {

	std::ofstream pixel_location_file;
	pixel_location_file.open(path + ".txt");

	double r_max = 	- std::numeric_limits<float>::infinity();
	double r_min =  std::numeric_limits<float>::infinity();


	for (unsigned int y_index = 0; y_index < this -> row_count; ++y_index) {

		if (std::abs(this -> focal_plane[y_index][0] -> get_range_residual()) < 1e10 )
			pixel_location_file << std::abs(this -> focal_plane[y_index][0] -> get_range_residual());
		else
			pixel_location_file << "nan";



		for (unsigned int z_index = 1; z_index < this -> col_count; ++z_index) {

			double range = std::abs(this -> focal_plane[y_index][z_index] -> get_range_residual()) ;


			if (std::abs(range) < 1e10 ) {

				pixel_location_file << " " << range ;
				if (range > r_max)
					r_max = range;
				if (range < r_min )
					r_min = range;
			}
			else
				pixel_location_file << " nan";


		}

		pixel_location_file << "\n";


	}

	pixel_location_file.close();

	return std::make_pair(r_min, r_max);

}


void Lidar::plot_ranges(std::string path, unsigned int type) const {

	std::pair<double, double> range_lims;
	switch (type) {
	case 0:
		range_lims = this -> save_true_range(path);
		break;
	case 1:
		range_lims = this -> save_computed_range(path);
		break;
	case 2:
		range_lims = this -> save_range_residuals(path);
		break;
	default:
		throw (std::runtime_error("Type not equal to 0, 1 or 2. Got " + std::to_string(type)));
		break;
	}

	std::vector<std::string> script;
	script.push_back("set terminal png");
	script.push_back("set output '" + path + ".png'");

	script.push_back("set title ''");
	script.push_back("set view map");
	script.push_back("set palette rgb 33, 13, 10");
	if (type != 2)
		script.push_back("set palette negative");
	script.push_back("set xrange [" + std::to_string(-1) + ":" + std::to_string(this -> col_count) + "]");
	script.push_back("set yrange [" + std::to_string(-1) + ":" + std::to_string(this -> row_count) + "]");
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

void Lidar::save_surface_measurements(std::string path) const {


	std::ofstream shape_file;
	shape_file.open(path);

	for (unsigned int vertex_index = 0;
	        vertex_index < this -> surface_measurements.size();
	        ++vertex_index) {

		shape_file << "v " << this -> surface_measurements[vertex_index](0) << " " << this -> surface_measurements[vertex_index](1) << " " << this -> surface_measurements[vertex_index](2) << std::endl;
	}



}




void Lidar::save_pixel_location(std::string path) const {

	std::ofstream pixel_location_file;
	pixel_location_file.open(path);

	// The focal plane is populated row by row
	for (unsigned int y_index = 0; y_index < this -> row_count; ++y_index) {


		// column by column
		for (unsigned int z_index = 0; z_index < this -> col_count; ++z_index) {

			pixel_location_file << this -> focal_plane[y_index][z_index] -> get_origin() -> colptr(0)[0] << " "
			                    << this -> focal_plane[y_index][z_index] -> get_origin() -> colptr(0)[1] << " "
			                    << this -> focal_plane[y_index][z_index] -> get_origin() -> colptr(0)[2] << "\n";

		}



	}
	pixel_location_file.close();

}
