#include "KDTree_pc.hpp"



KDTree_pc::KDTree_pc() {

}

void KDTree_pc::set_depth(int depth) {
	this -> depth = depth;
}



double KDTree_pc::get_value() const {
	return this -> value;
}

unsigned int KDTree_pc::get_axis() const {
	return this -> axis;
}

void KDTree_pc::set_value(double value) {
	this -> value = value;
}
void KDTree_pc::set_axis(unsigned int axis) {
	this -> axis = axis;
}

void KDTree_pc::closest_point_search(const arma::vec & test_point,
	std::shared_ptr<KDTree_pc> node,
	std::shared_ptr<PointNormal> & best_guess,
	double & distance) {



	if (node -> points_normals.size() == 1 ) {

		double new_distance = arma::norm(node -> points_normals[0] -> get_point() - test_point);

		if (new_distance < distance) {
			distance = new_distance;
			best_guess = node -> points_normals[0];
		}

	}

	else {

		bool search_left_first;



		if (test_point(node -> get_axis()) <= node -> get_value()) {
			search_left_first = true;
		}
		else {
			search_left_first = false;
		}

		if (search_left_first) {



			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> left,
					best_guess,
					distance);
			}

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess,
					distance);
			}

		}

		else {

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess,
					distance);
			}

			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> left,
					best_guess,
					distance);
			}

		}

	}


}


void KDTree_pc::closest_point_search(const arma::vec & test_point,
	std::shared_ptr<KDTree_pc> node,
	std::shared_ptr<PointNormal> & best_guess,
	double & distance,
	std::vector<std::shared_ptr<PointNormal> > & closest_points) {


	if (node -> points_normals.size() == 1 ) {

		double new_distance = arma::norm( node -> points_normals[0] -> get_point() - test_point);

		if (new_distance < distance && std::find(closest_points.begin(), closest_points.end(), node -> points_normals[0]) == closest_points.end()) {
			distance = new_distance;
			best_guess = node -> points_normals[0];
		}

	}

	else {

		bool search_left_first;


		if (test_point(node -> get_axis()) <= node -> get_value()) {
			search_left_first = true;
		}
		else {
			search_left_first = false;
		}
	
		if (search_left_first) {



			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> left,
					best_guess,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess,
					distance,
					closest_points);
			}

		}

		else {

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> left,
					best_guess,
					distance,
					closest_points);
			}

		}

	}


}



std::shared_ptr<KDTree_pc> KDTree_pc::build(std::vector< std::shared_ptr<PointNormal> > & points_normals, int depth, bool verbose) {


	// Creating the node
	std::shared_ptr<KDTree_pc> node = std::make_shared<KDTree_pc>(KDTree_pc());
	node -> points_normals = points_normals;
	node -> left = nullptr;
	node -> right = nullptr;
	node -> set_depth(depth);


	if (verbose) {
		std::cout << "Points in node: " << points_normals.size() <<  std::endl;
	}

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

		if (verbose) {
			std::cout << "Trivial node" << std::endl;
			std::cout << "Leaf depth: " << depth << std::endl;
		}

		return node;

	}

	arma::vec midpoint = arma::zeros<arma::vec>(3);
	arma::vec start_point = points_normals[0] -> get_point();

	arma::vec min_bounds = start_point;
	arma::vec max_bounds = start_point;


	// Could multithread here
	for (unsigned int i = 0; i < points_normals.size(); ++i) {

		arma::vec point = points_normals[i] -> get_point();

		max_bounds = arma::max(max_bounds,point);
		min_bounds = arma::min(min_bounds,point);

		// The midpoint of all the facets is found
		midpoint += (point / points_normals.size());
	}
	


	arma::vec bounding_box_lengths = max_bounds - min_bounds;

	// Facets to be assigned to the left and right nodes
	std::vector < std::shared_ptr<PointNormal> > left_points;
	std::vector < std::shared_ptr<PointNormal> > right_points;

	if (verbose){
		std::cout << "Midpoint: " << midpoint.t() << std::endl;
		std::cout << "Bounding box lengths: " << bounding_box_lengths.t();
	}


	if (arma::norm(bounding_box_lengths) == 0) {
		if (verbose) {
			std::cout << "Cluttered node" << std::endl;
		}
		// return node;
	}

	unsigned int longest_axis = bounding_box_lengths.index_max();

	if (longest_axis < 0 || longest_axis > 2){
		throw(std::runtime_error("overflow in longest_axis"));
	}

	for (unsigned int i = 0; i < points_normals.size() ; ++i) {

		if (midpoint(longest_axis) >= points_normals[i] -> get_point()(longest_axis)) {
			left_points.push_back(points_normals[i]);
		}

		else {
			right_points.push_back(points_normals[i]);
		}

	}

	node -> set_axis(longest_axis);
	node -> set_value(midpoint(longest_axis));

	// I guess this could be avoided
	if (left_points.size() == 0 && right_points.size() > 0) {
		left_points = right_points;
	}

	if (right_points.size() == 0 && left_points.size() > 0) {
		right_points = left_points;
	}

	

	// Recursion continues
	node -> left = build(left_points, depth + 1, verbose);
	node -> right = build(right_points, depth + 1, verbose);

	return node;

}

std::vector<std::shared_ptr<PointNormal> > * KDTree_pc::get_points_normals() {
	return &this -> points_normals;
}

unsigned int KDTree_pc::get_size() const {
	return this -> points_normals.size();
}


