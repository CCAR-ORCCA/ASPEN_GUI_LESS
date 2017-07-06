#include "PC.hpp"

PC::PC(arma::vec los_dir, std::vector<std::vector<std::shared_ptr<Ray> > > * focal_plane,
       FrameGraph * frame_graph) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	// The valid measurements used to form the point cloud are extracted
	for (unsigned int y_index = 0; y_index < focal_plane -> size(); ++y_index) {
		for (unsigned int z_index = 0; z_index < focal_plane -> at(0).size(); ++z_index) {

			if (focal_plane -> at(y_index)[z_index] -> get_true_range() < std::numeric_limits<double>::infinity()) {

				arma::vec impact_point_instrument = focal_plane -> at(y_index)[z_index] -> get_impact_point(false);

				arma::vec impact_point_inertial = frame_graph -> convert(impact_point_instrument, "L", "N");

				points_normals.push_back(std::make_shared<PointNormal>(PointNormal(impact_point_inertial)));
			}

		}
	}

	this -> construct_kd_tree(points_normals);
	this -> construct_normals(los_dir);

}



unsigned int PC::get_size() const {
	return this -> kd_tree -> points_normals.size();
}

arma::vec PC::get_point_coordinates(unsigned int index) const {
	return *this -> kd_tree -> points_normals[index] -> get_point();
}

std::shared_ptr<PointNormal> PC::get_point(unsigned int index) const {
	return this -> kd_tree -> points_normals[index];
}




arma::vec PC::get_point_normal(unsigned int index) const {
	return *this -> kd_tree -> points_normals[index] -> get_normal();
}






PC::PC(arma::vec los_dir, arma::mat & points) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	for (unsigned int index = 0; index < points . n_cols; ++index) {

		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(points . col(index))));
	}

	this -> construct_kd_tree(points_normals);
	this -> construct_normals(los_dir);


}



void PC::construct_kd_tree(std::vector< std::shared_ptr<PointNormal> > & points_normals) {

	// The KD Tree is now constructed
	this -> kd_tree = std::make_shared<KDTree_pc>(KDTree_pc());
	this -> kd_tree = this -> kd_tree -> build(points_normals, 0, false);

}

std::shared_ptr<PointNormal> PC::get_closest_point(arma::vec & test_point) const {

	double distance = std::numeric_limits<double>::infinity();
	std::shared_ptr<PointNormal> closest_point;

	this -> kd_tree -> closest_point_search(test_point,
	                                        this -> kd_tree,
	                                        closest_point,
	                                        distance);

	return closest_point;

}

std::vector<std::shared_ptr<PointNormal> > PC::get_closest_N_points(
    arma::vec & test_point, unsigned int N) const {

	std::vector<std::shared_ptr<PointNormal> > closest_points;

	for (unsigned int i = 0; i < N; ++i) {

		double distance = std::numeric_limits<double>::infinity();
		std::shared_ptr<PointNormal> closest_point;

		this -> kd_tree -> closest_point_search(test_point,
		                                        this -> kd_tree,
		                                        closest_point,
		                                        distance,
		                                        closest_points);

		closest_points.push_back(closest_point);
	}

	return closest_points;

}

void PC::save(std::string path) const {



	std::ofstream shape_file;
	shape_file.open(path);

	for (unsigned int vertex_index = 0;
	        vertex_index < this -> get_size();
	        ++vertex_index) {

		arma::vec p = this -> get_point_coordinates(vertex_index);
		shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
	}



}


void  PC::save(std::string path, arma::mat dcm, arma::vec x) const {


	std::ofstream shape_file;
	shape_file.open(path);

	for (unsigned int vertex_index = 0;
	        vertex_index < this -> get_size();
	        ++vertex_index) {

		arma::vec p = dcm * this -> get_point_coordinates(vertex_index) + x;
		shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
	}


}




std::shared_ptr<PointNormal> PC::get_closest_point_index_brute_force(arma::vec & test_point) const {

	double distance = std::numeric_limits<double>::infinity();

	unsigned int pc_size = this -> kd_tree -> get_size();
	int index_closest = 0;

	for (unsigned int i = 0 ; i < pc_size ; ++i) {
		double new_distance = arma::norm(test_point - *this -> kd_tree ->
		                                 get_points_normals() -> at(i) -> get_point());

		if (new_distance < distance) {
			index_closest = i;
			distance = new_distance;
		}

	}

	return this -> kd_tree -> get_points_normals() -> at(index_closest);
}

std::vector<std::shared_ptr<PointNormal> > PC::get_closest_N_points_brute_force(arma::vec & test_point, unsigned int N) const {

	unsigned int pc_size = this -> kd_tree -> get_size();

	if (N > pc_size) {
		throw "Point cloud has fewer points than requested neighborhood size";
	}

	if (pc_size == 0) {
		throw "Empty point cloud";
	}

	std::map<double, unsigned int> distance_map;
	distance_map[std::numeric_limits<double>::infinity()] = 0;

	for (unsigned int i = 0 ; i < pc_size ; ++i) {

		double new_distance = arma::norm(test_point - *this -> kd_tree ->
		                                 get_points_normals() -> at(i) -> get_point());

		if (new_distance > std::prev(distance_map.end()) -> first) {
			continue;
		}

		else  {

			distance_map[new_distance] = i;

			if (distance_map.size() > N) {
				distance_map.erase(std::prev(distance_map.end()));
			}

		}

	}

	std::vector<std::shared_ptr<PointNormal> > closest_points;

	for (auto it = distance_map.begin(); it != distance_map.end(); ++it ) {
		closest_points.push_back(this -> kd_tree -> get_points_normals() -> at(it -> second));
	}

	return closest_points;
}


void PC::construct_normals(arma::vec & los_dir) {

	#pragma omp parallel for
	for (unsigned int i = 0; i < this -> kd_tree -> get_points_normals() -> size(); ++i) {

		std::shared_ptr<PointNormal> pn = this -> kd_tree -> get_points_normals() -> at(i);

		// Get the N nearest neighbors to this point
		unsigned int N = 5;
		std::vector< std::shared_ptr<PointNormal > > closest_points = this -> get_closest_N_points(*pn -> get_point(), N);

		// This N nearest neighbors are used to get the normal
		arma::mat points_augmented(N, 4);
		points_augmented.col(3) = arma::ones<arma::vec>(N);

		for (unsigned int j = 0; j < N; ++j) {
			points_augmented.row(j).cols(0, 2) = closest_points[j] -> get_point() -> t();
		}

		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, points_augmented.t() * points_augmented);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		if (arma::dot(n, los_dir) < 0) {
			this -> kd_tree -> get_points_normals() -> at(i) -> set_normal(n);
		}
		else {
			this -> kd_tree -> get_points_normals() -> at(i) -> set_normal(-n);
		}

	}

}


