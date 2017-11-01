#include "KDTree_facet.hpp"



KDTree_facet::KDTree_facet() {

}

void KDTree_facet::set_depth(int depth) {
	this -> depth = depth;
}



double KDTree_facet::get_value() const {
	return this -> value;
}

unsigned int KDTree_facet::get_axis() const {
	return this -> axis;
}

void KDTree_facet::set_value(double value) {
	this -> value = value;
}
void KDTree_facet::set_axis(unsigned int axis) {
	this -> axis = axis;
}

void KDTree_facet::closest_point_search(const arma::vec & test_point,
                                     std::shared_ptr<KDTree_facet> node,
                                     std::shared_ptr<Element> & best_guess,
                                     double & distance) {



	if (node -> facets.size() == 1 ) {

		double new_distance = arma::norm(* dynamic_cast<Facet *>(node -> facets[0].get()) -> get_facet_center() - test_point);

		if (new_distance < distance) {
			distance = new_distance;
			best_guess = node -> facets[0];
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


void KDTree_facet::closest_point_search(const arma::vec & test_point,
                                     std::shared_ptr<KDTree_facet> node,
                                     std::shared_ptr<Element> & best_guess,
                                     double & distance,
                                     std::vector<std::shared_ptr<Element> > & closest_points) {



	if (node -> facets.size() == 1 ) {

		double new_distance = arma::norm(* dynamic_cast<Facet *>(node -> facets[0].get()) -> get_facet_center() - test_point);

		if (new_distance < distance && std::find(closest_points.begin(), closest_points.end(), node -> facets[0]) == closest_points.end()) {
			distance = new_distance;
			best_guess = node -> facets[0];
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



std::shared_ptr<KDTree_facet> KDTree_facet::build(std::vector< std::shared_ptr<Element> > & facets, int depth, bool verbose) {

	// Creating the node
	std::shared_ptr<KDTree_facet> node = std::make_shared<KDTree_facet>(KDTree_facet());
	node -> facets = facets;
	node -> left = nullptr;
	node -> right = nullptr;
	node -> set_depth(depth);

	if (facets.size() == 0) {
		if (verbose) {
			std::cout << "Empty node" << std::endl;
			std::cout << "Leaf depth: " << depth << std::endl;
		}
		return node;
	}


	// If the node only contains one triangle,
	// there's no point in subdividing it more
	if (facets.size() == 1) {

		node -> left = std::make_shared<KDTree_facet>( KDTree_facet() );
		node -> right = std::make_shared<KDTree_facet>( KDTree_facet() );

		node -> left -> facets = std::vector<std::shared_ptr<Element> >();
		node -> right -> facets = std::vector<std::shared_ptr<Element> >();

		if (verbose) {
			std::cout << "Trivial node" << std::endl;
			std::cout << "Leaf depth: " << depth << std::endl;
		}

		return node;

	}

	arma::vec midpoint = arma::zeros<arma::vec>(3);

	double xmin = dynamic_cast<Facet *>(facets[0].get()) -> get_facet_center() -> at(0);
	double xmax = dynamic_cast<Facet *>(facets[0].get()) -> get_facet_center() -> at(0);

	double ymin = dynamic_cast<Facet *>(facets[0].get()) -> get_facet_center() -> at(1);
	double ymax = dynamic_cast<Facet *>(facets[0].get()) -> get_facet_center() -> at(1);

	double zmin = dynamic_cast<Facet *>(facets[0].get()) -> get_facet_center() -> at(2);
	double zmax = dynamic_cast<Facet *>(facets[0].get()) -> get_facet_center() -> at(2);


	// Could multithread here
	for (unsigned int i = 0; i < facets.size(); ++i) {

		xmin = std::min(dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center() -> at(0), xmin);
		xmax = std::max(dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center() -> at(0), xmax);

		ymin = std::min(dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center() -> at(1), ymin);
		ymax = std::max(dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center() -> at(1), ymax);

		zmin = std::min(dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center() -> at(2), zmin);
		zmax = std::max(dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center() -> at(2), zmax);


		// The midpoint of all the facets is found
		midpoint += (*dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center()) * (1. / facets.size());
	}

	arma::vec bounding_box_lengths = {xmax - xmin, ymax - ymin, zmax - zmin};

	// Elements to be assigned to the left and right nodes
	std::vector < std::shared_ptr<Element> > left_points;
	std::vector < std::shared_ptr<Element> > right_points;

	unsigned int longest_axis = bounding_box_lengths.index_max();

	for (unsigned int i = 0; i < facets.size() ; ++i) {

		if (midpoint(longest_axis) >= dynamic_cast<Facet *>(facets[i].get()) -> get_facet_center() -> at(longest_axis)) {
			left_points.push_back(facets[i]);
		}

		else {
			right_points.push_back(facets[i]);
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

std::vector<std::shared_ptr<Element> > * KDTree_facet::get_facets() {
	return &this -> facets;
}

unsigned int KDTree_facet::get_size() const {
	return this -> facets.size();
}


