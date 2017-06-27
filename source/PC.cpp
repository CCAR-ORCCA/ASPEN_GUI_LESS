#include "PC.hpp"

PC::PC(arma::vec los_dir, std::vector<std::vector<std::shared_ptr<Ray> > > * focal_plane) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	// The valid measurements used to form the point cloud are extracted
	for (unsigned int y_index = 0; y_index < focal_plane -> size(); ++y_index) {
		for (unsigned int z_index = 0; z_index < focal_plane -> at(0).size(); ++z_index) {

			if (focal_plane -> at(y_index)[z_index] -> get_true_range() < std::numeric_limits<double>::infinity()) {
				points_normals.push_back(std::make_shared<PointNormal>(PointNormal(focal_plane -> at(y_index)[z_index] -> get_impact_point(true))));
			}

		}
	}

	this -> construct_kd_tree(points_normals);

}


PC::PC(arma::vec los_dir, arma::mat & points) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	for (unsigned int index = 0; index < points . n_cols; ++index) {
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(points . col(index))));
	}

	this -> construct_kd_tree(points_normals);

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



int PC::get_closest_point_index_brute_force(arma::vec & test_point) const {

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

	return index_closest;
}

arma::uvec PC::get_closest_points_indices_brute_force(arma::vec & test_point, unsigned int N) const {

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

	arma::uvec closest_indices(N);

	for (auto it = distance_map.begin(); it != distance_map.end(); ++it ) {
		closest_indices(std::distance(distance_map.begin(), it)) = it -> second;
	}

	return closest_indices;
}

