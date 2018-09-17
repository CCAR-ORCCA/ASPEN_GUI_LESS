#include "KDTreeDescriptors.hpp"
#include "DebugFlags.hpp"

#include <armadillo>


#define KDTTREE_DESCRIPTOR_DEBUG 0

KDTreeDescriptors::KDTreeDescriptors() {

}

void KDTreeDescriptors::set_depth(int depth) {
	this -> depth = depth;
}


double KDTreeDescriptors::get_value() const {
	return this -> value;
}

unsigned int KDTreeDescriptors::get_axis() const {
	return this -> axis;
}

void KDTreeDescriptors::set_value(double value) {
	this -> value = value;
}
void KDTreeDescriptors::set_axis(unsigned int axis) {
	this -> axis = axis;
}

void KDTreeDescriptors::set_is_cluttered(bool cluttered){
	this -> cluttered = cluttered;
}

bool KDTreeDescriptors::get_is_cluttered() const{
	return this -> cluttered;
}


void KDTreeDescriptors::closest_point_search(std::shared_ptr<PointNormal> query_point,
	std::shared_ptr<KDTreeDescriptors> node,
	std::shared_ptr<PointNormal> & best_guess,
	double & distance) {


		#if KDTTREE_DESCRIPTOR_DEBUG
	std::cout << "#############################\n";
	std::cout << "Depth: " << this -> depth << std::endl; ;
	std::cout << "Points in node: " << node -> points_with_descriptors.size() << std::endl;
	std::cout << "Best distance found so far : " << distance << std::endl;
		#endif

	if (node -> points_with_descriptors.size() == 1 || node -> get_is_cluttered()) {


		#if KDTTREE_DESCRIPTOR_DEBUG
		std::cout << "Leaf node\n";
		#endif

		double new_distance = node -> points_with_descriptors[0] -> descriptor_distance(query_point);
		
		#if KDTTREE_DESCRIPTOR_DEBUG
		std::cout << "Distance to query_point: " << new_distance << std::endl;
		#endif

		if (new_distance < distance) {
			distance = new_distance;
			best_guess = node -> points_with_descriptors[0];

			#if KDTTREE_DESCRIPTOR_DEBUG
			std::cout << "Found closest point " << best_guess << " with distance = " + std::to_string(distance)<< " \n" << std::endl;
			#endif
		}

	}

	else {
		bool search_left_first;
		#if KDTTREE_DESCRIPTOR_DEBUG
		std::cout << "Fork node\n";
		#endif

		if (query_point -> get_histogram_value(node -> get_axis()) <= node -> get_value()) {
			

			search_left_first = true;
		}
		else {
			search_left_first = false;
		}

		if (search_left_first) {

			#if KDTTREE_DESCRIPTOR_DEBUG
			std::cout << "Searching left first\n";
			#endif

			if (query_point -> get_histogram_value(node -> get_axis()) - distance <= node -> get_value()) {
				
				node -> closest_point_search(query_point,
					node -> left,
					best_guess,
					distance);
			}

			if (query_point -> get_histogram_value(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(query_point,
					node -> right,
					best_guess,
					distance);
			}

		}

		else {

			#if KDTTREE_DESCRIPTOR_DEBUG
			std::cout << "Searching right first\n";
			#endif

			if (query_point -> get_histogram_value(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(query_point,
					node -> right,
					best_guess,
					distance);
			}

			if (query_point -> get_histogram_value(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(query_point,
					node -> left,
					best_guess,
					distance);
			}

		}

	}


}


std::shared_ptr<KDTreeDescriptors> KDTreeDescriptors::build(std::vector< std::shared_ptr<PointNormal> > & points_with_descriptors, int depth) {


	// Creating the node
	std::shared_ptr<KDTreeDescriptors> node = std::make_shared<KDTreeDescriptors>(KDTreeDescriptors());
	node -> points_with_descriptors = points_with_descriptors;
	node -> left = nullptr;
	node -> right = nullptr;
	node -> set_depth(depth);


	#if KDTTREE_DESCRIPTOR_DEBUG
	std::cout << "#############################\n";
	std::cout << "Depth: " << depth << std::endl; ;
	std::cout << "Points in node: " << points_with_descriptors.size() <<  std::endl;
	#endif

	if (points_with_descriptors.size() == 0) {
		#if KDTTREE_DESCRIPTOR_DEBUG
		std::cout << "Empty node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;

		#endif
		return node;
	}


	// If the node only contains one triangle,
	// there's no point in subdividing it more
	if (points_with_descriptors.size() == 1) {

		node -> left = std::make_shared<KDTreeDescriptors>( KDTreeDescriptors() );
		node -> right = std::make_shared<KDTreeDescriptors>( KDTreeDescriptors() );

		node -> left -> points_with_descriptors = std::vector< std::shared_ptr<PointNormal> >();
		node -> right -> points_with_descriptors = std::vector< std::shared_ptr<PointNormal> >();

		#if KDTTREE_DESCRIPTOR_DEBUG
		std::cout << "Trivial node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif

		return node;

	}

	arma::vec midpoint = arma::zeros<arma::vec>(points_with_descriptors[0] -> get_histogram_size());
	arma::vec start_point(points_with_descriptors[0] -> get_histogram_size());

	for (int k =0; k < midpoint.n_rows; ++k){
		start_point(k) =  points_with_descriptors[0] -> get_histogram_value(k);
	}


	arma::vec min_bounds = start_point;
	arma::vec max_bounds = start_point;

	// Could multithread here
	for (unsigned int i = 0; i < points_with_descriptors.size(); ++i) {

		std::shared_ptr<PointNormal> point_with_descriptor = points_with_descriptors[i];

		for (int k =0; k < midpoint.n_rows; ++k){
			double hist_value = point_with_descriptor -> get_histogram_value(k);
			max_bounds(k) = std::max(max_bounds(k),hist_value);
			min_bounds(k) = std::min(min_bounds(k),hist_value);

			midpoint(k) += (hist_value / points_with_descriptors.size());

		}

	}
	


	arma::vec bounding_box_lengths = max_bounds - min_bounds;

	// Facets to be assigned to the left and right nodes
	std::vector < std::shared_ptr<PointNormal> > left_points;
	std::vector < std::shared_ptr<PointNormal> > right_points;

	#if KDTTREE_DESCRIPTOR_DEBUG
	std::cout << "Midpoint: " << midpoint.t() << std::endl;
	std::cout << "Bounding box lengths: " << bounding_box_lengths.t();
	#endif


	if (arma::norm(bounding_box_lengths) < 1e-15) {
		
		#if KDTTREE_DESCRIPTOR_DEBUG
		std::cout << "Cluttered node" << std::endl;
		#endif

		node -> set_is_cluttered(true);
		
		return node;
	}

	int longest_axis = bounding_box_lengths.index_max();


	for (unsigned int i = 0; i < points_with_descriptors.size() ; ++i) {

		if (midpoint(longest_axis) >= points_with_descriptors[i] -> get_histogram_value(longest_axis)) {
			left_points.push_back(points_with_descriptors[i]);
		}

		else {
			right_points.push_back(points_with_descriptors[i]);
		}

	}

	#if KDTTREE_DESCRIPTOR_DEBUG
	std::cout << "Split axis : " << longest_axis << std::endl;
	std::cout << "Split value : " << midpoint(longest_axis) << std::endl;
	#endif

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
	node -> left = build(left_points, depth + 1);
	node -> right = build(right_points, depth + 1);

	return node;

}

unsigned int KDTreeDescriptors::get_size() const {
	return this -> points_with_descriptors.size();
}


