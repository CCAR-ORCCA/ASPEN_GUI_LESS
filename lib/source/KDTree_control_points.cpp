#include "KDTree_control_points.hpp"



KDTree_control_points::KDTree_control_points() {

}

void KDTree_control_points::set_depth(int depth) {
	this -> depth = depth;
}



double KDTree_control_points::get_value() const {
	return this -> value;
}

unsigned int KDTree_control_points::get_axis() const {
	return this -> axis;
}

void KDTree_control_points::set_value(double value) {
	this -> value = value;
}
void KDTree_control_points::set_axis(unsigned int axis) {
	this -> axis = axis;
}

void KDTree_control_points::closest_point_search(const arma::vec & test_point,
	std::shared_ptr<KDTree_control_points> node,
	std::shared_ptr<ControlPoint> & best_guess,
	double & distance) {



	if (node -> control_points.size() == 1 ) {

		double new_distance = arma::norm( node -> control_points[0] -> get_coordinates() - test_point);

		if (new_distance < distance) {
			distance = new_distance;
			best_guess = node -> control_points[0];
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


void KDTree_control_points::closest_point_search(const arma::vec & test_point,
	std::shared_ptr<KDTree_control_points> node,
	std::shared_ptr<ControlPoint> & best_guess,
	double & distance,
	std::vector<std::shared_ptr<ControlPoint> > & closest_points) {



	if (node -> control_points.size() == 1 ) {

		double new_distance = arma::norm( node -> control_points[0] -> get_coordinates() - test_point);

		if (new_distance < distance && std::find(closest_points.begin(), closest_points.end(), node -> control_points[0]) == closest_points.end()) {
			distance = new_distance;
			best_guess = node -> control_points[0];
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

std::shared_ptr<KDTree_control_points> KDTree_control_points::build(std::vector< std::shared_ptr<ControlPoint> > & control_points, int depth, bool verbose) {

	// Creating the node
	std::shared_ptr<KDTree_control_points> node = std::make_shared<KDTree_control_points>(KDTree_control_points());
	node -> control_points = control_points;
	node -> left = nullptr;
	node -> right = nullptr;
	node -> set_depth(depth);

	if (verbose) {
		std::cout << "Points in node: " << control_points.size() <<  std::endl;
	}

	if (control_points.size() == 0) {
		if (verbose) {
			std::cout << "Empty node" << std::endl;
			std::cout << "Leaf depth: " << depth << std::endl;
		}
		return node;
	}


	// If the node only contains one point,
	// there's no point in subdividing it more
	if (control_points.size() == 1) {

		node -> left = std::make_shared<KDTree_control_points>( KDTree_control_points() );
		node -> right = std::make_shared<KDTree_control_points>( KDTree_control_points() );

		node -> left -> control_points = std::vector<std::shared_ptr<ControlPoint> >();
		node -> right -> control_points = std::vector<std::shared_ptr<ControlPoint> >();

		if (verbose) {
			std::cout << "Trivial node" << std::endl;
			std::cout << "Leaf depth: " << depth << std::endl;
		}

		return node;

	}

	arma::vec midpoint = arma::zeros<arma::vec>(3);
	arma::vec start_point = control_points[0] -> get_coordinates();

	arma::vec min_bounds = start_point;
	arma::vec max_bounds = start_point;

	// Could multithread here
	for (unsigned int i = 0; i < control_points.size(); ++i) {

		arma::vec point = control_points[i] -> get_coordinates();

		max_bounds = arma::max(max_bounds,point);
		min_bounds = arma::min(min_bounds,point);

		// The midpoint of all the facets is found
		midpoint += (control_points[i] -> get_coordinates()/ control_points.size());
	}
	arma::vec bounding_box_lengths = max_bounds - min_bounds;

	if (verbose){
		std::cout << "Midpoint: " << midpoint.t() << std::endl;
		std::cout << "Bounding box lengths: " << bounding_box_lengths.t();
	}


	if (arma::norm(bounding_box_lengths) == 0) {
		if (verbose) {
			std::cout << "Cluttered node" << std::endl;
		}
		return node;
	}

	// Facets to be assigned to the left and right nodes
	std::vector < std::shared_ptr<ControlPoint> > left_points;
	std::vector < std::shared_ptr<ControlPoint> > right_points;

	unsigned int longest_axis = bounding_box_lengths.index_max();

	for (unsigned int i = 0; i < control_points.size() ; ++i) {

		if (midpoint(longest_axis) >= control_points[i] -> get_coordinates()(longest_axis)) {
			left_points.push_back(control_points[i]);
		}

		else {
			right_points.push_back(control_points[i]);
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

std::vector<std::shared_ptr<ControlPoint> > * KDTree_control_points::get_control_points() {
	return &this -> control_points;
}

unsigned int KDTree_control_points::get_size() const {
	return this -> control_points.size();
}


