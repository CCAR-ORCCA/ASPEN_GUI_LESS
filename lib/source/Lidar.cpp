#include "Lidar.hpp"
#include "ShapeModelBezier.hpp"

Lidar::Lidar(
	FrameGraph * frame_graph,
	std::string ref_frame_name,
	double fov_y,
	double fov_z,
	unsigned int y_res,
	unsigned int z_res,
	double f,
	double freq,
	double los_noise_3sd_baseline,
	double los_noise_fraction_mes_truth) {

	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
	this -> f = f;
	this -> fov_y = fov_y;
	this -> fov_z = fov_z;
	this -> z_res = z_res;
	this -> y_res = y_res;
	this -> freq = freq;
	this -> los_noise_sd_baseline = los_noise_3sd_baseline / 3;
	this -> los_noise_fraction_mes_truth = los_noise_fraction_mes_truth;

	// The focal plane of the lidar is populated
	for (unsigned int z_index = 0; z_index < this -> z_res; ++z_index) {	
		for (unsigned int y_index = 0; y_index < this -> y_res; ++y_index) {
			this -> focal_plane.push_back(std::make_shared<Ray>(Ray(y_index, z_index, this)));
		}

	}


}

double Lidar::get_z_res() const {
	return (double)(this -> z_res);
}

double Lidar::get_y_res() const {
	return (double)(this -> y_res);
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


void Lidar::send_flash(ShapeModel * shape_model,bool add_noise) {

	unsigned int y_res = this -> z_res;
	unsigned int z_res = this -> y_res;

	
	for (unsigned int pixel = 0; pixel < y_res * z_res; ++pixel){
		this -> focal_plane[pixel] -> reset( shape_model);
	}
	

	#pragma omp parallel for if (USE_OMP_LIDAR)
	for (unsigned int pixel = 0; pixel < y_res * z_res; ++pixel){

		bool hit = shape_model -> ray_trace(this -> focal_plane[pixel].get());

		// If there's a hit, noise is added along the line of sight on the true measurement
		if (hit) {

			double true_range = this -> focal_plane[pixel] -> get_true_range();

			if (add_noise){
				arma::vec random_vec = arma::randn(1);

				double noise_sd = this -> los_noise_sd_baseline + this -> los_noise_fraction_mes_truth * true_range;			
				double noise = noise_sd * random_vec(0);

				this -> focal_plane[pixel] -> set_true_range(true_range + noise);
			}
			else{
				this -> focal_plane[pixel] -> set_true_range(true_range);

			}
			
		}

	}

}




ShapeModel * Lidar::get_shape_model() {
	return this -> shape_model;
}

std::string Lidar::get_ref_frame_name() const {
	return this -> ref_frame_name;
}

std::vector<std::shared_ptr<Ray> > * Lidar::get_focal_plane() {
	return &this -> focal_plane;
}



Ray * Lidar::get_ray(unsigned int pixel) {
	return this -> focal_plane[pixel].get();
}


unsigned int Lidar::get_number_of_hits() const{

	unsigned int hit = 0;
	
	#pragma omp parallel for reduction(+:hit)
	for (unsigned int pixel = 0; pixel < this -> focal_plane . size(); ++pixel) {

		if (this ->  focal_plane[pixel] -> get_true_range() < std::numeric_limits<double>::infinity()) {
			++hit;
		}
	}

	return hit;

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


void Lidar::save(std::string path) {


	std::ofstream shape_file;
	shape_file.open(path);

	for (unsigned int i = 0;i < this -> focal_plane.size();++i) {

		Ray  * ray = this -> get_ray(i);

		if (ray-> get_hit_element() != nullptr){
			arma::vec p = (*ray-> get_direction_target_frame()) * ray-> get_true_range() + *ray-> get_origin_target_frame();
			shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
		}

	}



}




