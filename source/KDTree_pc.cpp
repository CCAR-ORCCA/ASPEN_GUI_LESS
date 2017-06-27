#include "KDTree_pc.hpp"



KDTree_pc::KDTree_pc() {

}

void KDTree_pc::set_depth(int depth) {
	this -> depth = depth;
}

std::shared_ptr<KDTree_pc> KDTree_pc::build(std::vector< std::shared_ptr<PointNormal> > & points_normals, int depth, bool verbose) {

	// Creating the node
	std::shared_ptr<KDTree_pc> node = std::make_shared<KDTree_pc>(KDTree_pc());
	node -> points_normals = points_normals;
	node -> left = nullptr;
	node -> right = nullptr;
	node -> set_depth(depth);

	if (points_normals.size() == 0) {
		if (verbose) {
			std::cout << "Empty node" << std::endl;
			std::cout << "Leaf depth: " << depth << std::endl;
		}
		return node;
	}


	// If the node only contains one triangle,
	// there's no point in subdividing it more
	if (points_normals.size() == 1) {

		node -> left = std::make_shared<KDTree_pc>( KDTree_pc() );
		node -> right = std::make_shared<KDTree_pc>( KDTree_pc() );

		node -> left -> points_normals = std::vector<std::shared_ptr<PointNormal> >();
		node -> right -> points_normals = std::vector<std::shared_ptr<PointNormal> >();


		return node;

	}

	arma::vec midpoint = arma::zeros<arma::vec>(3);

	double xmin = points_normals[0] -> get_point() -> at(0);
	double xmax = points_normals[0] -> get_point() -> at(0);

	double ymin = points_normals[0] -> get_point() -> at(1);
	double ymax = points_normals[0] -> get_point() -> at(1);

	double zmin = points_normals[0] -> get_point() -> at(2);
	double zmax = points_normals[0] -> get_point() -> at(2);



	// Could multithread here
	for (unsigned int i = 0; i < points_normals.size(); ++i) {

		xmin = std::min(points_normals[i] -> get_point() -> at(0), xmin);
		xmax = std::min(points_normals[i] -> get_point() -> at(0), xmax);

		ymax = std::min(points_normals[i] -> get_point() -> at(1), ymax);
		ymin = std::min(points_normals[i] -> get_point() -> at(1), ymax);

		zmax = std::min(points_normals[i] -> get_point() -> at(2), zmax);
		zmin = std::min(points_normals[i] -> get_point() -> at(2), zmax);


		// The midpoint of all the facets is found
		midpoint += (*points_normals[i] -> get_point()) * (1. / points_normals.size());
	}

	arma::vec bounding_box_lengths = {xmax - xmin, ymax - ymin, zmax - zmin};

	// Facets to be assigned to the left and right nodes
	std::vector < std::shared_ptr<PointNormal> > left_points;
	std::vector < std::shared_ptr<PointNormal> > right_points;

	unsigned int longest_axis = bounding_box_lengths.index_max();

	for (unsigned int i = 0; i < points_normals.size() ; ++i) {

		if (midpoint(longest_axis) >= points_normals[i] -> get_point() -> at(longest_axis)) {
			right_points.push_back(points_normals[i]);
		}

		else {
			left_points.push_back(points_normals[i]);
		}

	}

	// I guess this could be avoided
	if (left_points.size() == 0 && right_points.size() > 0) {
		left_points = right_points;
	}

	if (right_points.size() == 0 && left_points.size() > 0) {
		right_points = left_points;
	}

	// Recursion continues
	node -> left = build(left_points, depth + 1);
	node -> right = build(right_points, depth + 1);

	return node;

}

std::vector<std::shared_ptr<PointNormal> > * KDTree_pc::get_points_normals() {
	return &this -> points_normals;
}

unsigned int KDTree_pc::get_size() const {
	return this -> points_normals.size();
}


