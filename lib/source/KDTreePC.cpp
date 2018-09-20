#include "KDTreePC.hpp"
#include "DebugFlags.hpp"

#define KDTTREE_PC_DEBUG 0

KDTreePC::KDTreePC() {

}

void KDTreePC::set_depth(int depth) {
	this -> depth = depth;
}


double KDTreePC::get_value() const {
	return this -> value;
}

unsigned int KDTreePC::get_axis() const {
	return this -> axis;
}

void KDTreePC::set_value(double value) {
	this -> value = value;
}
void KDTreePC::set_axis(unsigned int axis) {
	this -> axis = axis;
}


void KDTreePC::set_is_cluttered(bool cluttered){
	this -> cluttered = cluttered;
}

bool KDTreePC::get_is_cluttered() const{
	return this -> cluttered;
}


void KDTreePC::closest_point_search(const arma::vec & test_point,
	std::shared_ptr<KDTreePC> node,
	std::shared_ptr<PointNormal> & best_guess,
	double & distance) {


	if (node -> points_normals.size() == 1 || node -> get_is_cluttered() ) {

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

void KDTreePC::closest_N_point_search(const arma::vec & test_point,
	const unsigned int & N_points,
	std::shared_ptr<KDTreePC> node,
	double & distance,
	std::map<double,std::shared_ptr<PointNormal> > & closest_points) {

	#if KDTTREE_PC_DEBUG
	std::cout << "#############################\n";
	std::cout << "Depth: " << this -> depth << std::endl; ;
	std::cout << "Points in node: " << node -> points_normals.size() << std::endl;
	std::cout << "Points found so far : " << closest_points.size() << std::endl;
	#endif

	// DEPRECATED
	if (node -> points_normals.size() == 1 || node -> get_is_cluttered()) {

		#if KDTTREE_PC_DEBUG
		std::cout << "Leaf node\n";
		#endif

		
		double new_distance = arma::norm( node -> points_normals[0] -> get_point() - test_point);

		#if KDTTREE_PC_DEBUG
		std::cout << "Distance to query_point: " << new_distance << std::endl;
		#endif

		if (closest_points.size() < N_points){
			closest_points[new_distance] = node -> points_normals[0];
		}
		else{

			unsigned int size_before = closest_points.size(); // should always be equal to N_points

			closest_points[new_distance] = node -> points_normals[0];

			unsigned int size_after = closest_points.size(); // should always be equal to N_points + 1, unless new_distance was already in the map

			if (size_after == size_before + 1){

				// Remove last element in map
				closest_points.erase(--closest_points.end());

				// Set the distance to that between the query point and the last element in the map
				distance = (--closest_points.end()) -> first;

				
			}


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
				node -> closest_N_point_search(test_point,
					N_points,
					node -> left,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_N_point_search(test_point,
					N_points,
					node -> right,
					distance,
					closest_points);
			}

		}

		else {

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_N_point_search(test_point,
					N_points,
					node -> right,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_N_point_search(test_point,
					N_points,
					node -> left,
					distance,
					closest_points);
			}

		}

	}

	#if KDTTREE_PC_DEBUG
	std::cout << "#############################\n";
	std::cout << " Leaving "<<  std::endl; ;
	std::cout << "Points found : " << std::endl;
	for (auto it = closest_points.begin(); it != closest_points.end(); ++it){
		std::cout << it -> first << " : " << it -> second -> get_point().t();
	}
	#endif


}


void KDTreePC::radius_point_search(const arma::vec & test_point,
	std::shared_ptr<KDTreePC> node,
	const double & distance,
	std::vector<std::shared_ptr<PointNormal> > & closest_points) {


	#if KDTTREE_PC_DEBUG
	std::cout << "#############################\n";
	std::cout << "Depth: " << this -> depth << std::endl; ;
	std::cout << "Points in node: " << node -> points_normals.size() << std::endl;
	std::cout << "Points found so far : " << closest_points.size() << std::endl;
	#endif




	if (node -> points_normals.size() == 1 || node -> get_is_cluttered() ) {

		#if KDTTREE_PC_DEBUG
		std::cout << "Leaf node\n";
		#endif

		double new_distance = arma::norm( node -> points_normals[0] -> get_point() - test_point);

		#if KDTTREE_PC_DEBUG
		std::cout << "Distance to query_point: " << new_distance << std::endl;
		#endif

		if (new_distance < distance ) {
			// Takes care of cluttered nodes as well
			for (int i = 0; i < node -> points_normals.size(); ++i){
				closest_points.push_back(node -> points_normals[i]);
			}
			#if KDTTREE_PC_DEBUG
			std::cout << "Found closest point " << node -> points_normals[0] << " with distance = " + std::to_string(distance)<< " \n" << std::endl;
			#endif
		}

	}

	else {

		bool search_left_first;
		#if KDTTREE_PC_DEBUG
		std::cout << "Fork node\n";
		#endif

		if (test_point(node -> get_axis()) <= node -> get_value()) {
			search_left_first = true;
		}
		else {
			search_left_first = false;
		}

		if (search_left_first) {

			#if KDTTREE_PC_DEBUG
			std::cout << "Searching left first\n";
			#endif


			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> radius_point_search(test_point,
					node -> left,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> radius_point_search(test_point,
					node -> right,
					distance,
					closest_points);
			}

		}

		else {

			#if KDTTREE_PC_DEBUG
			std::cout << "Searching right first\n";
			#endif
			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> radius_point_search(test_point,
					node -> right,
					distance,
					closest_points);
			}

			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> radius_point_search(test_point,
					node -> left,
					distance,
					closest_points);
			}

		}

	}


}



std::shared_ptr<KDTreePC> KDTreePC::build(std::vector< std::shared_ptr<PointNormal> > & points_normals, int depth) {


	// Creating the node
	std::shared_ptr<KDTreePC> node = std::make_shared<KDTreePC>(KDTreePC());
	node -> points_normals = points_normals;
	node -> left = nullptr;
	node -> right = nullptr;
	node -> set_depth(depth);


	#if KDTTREE_PC_DEBUG
	std::cout << "Points in node: " << points_normals.size() <<  std::endl;
	#endif

	if (points_normals.size() == 0) {
		#if KDTTREE_PC_DEBUG
		std::cout << "Empty node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif
		return node;
	}


	// If the node only contains one triangle,
	// there's no point in subdividing it more
	if (points_normals.size() == 1) {

		node -> left = std::make_shared<KDTreePC>( KDTreePC() );
		node -> right = std::make_shared<KDTreePC>( KDTreePC() );

		node -> left -> points_normals = std::vector<std::shared_ptr<PointNormal> >();
		node -> right -> points_normals = std::vector<std::shared_ptr<PointNormal> >();

		#if KDTTREE_PC_DEBUG
		std::cout << "Trivial node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif

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

	#if KDTTREE_PC_DEBUG
	std::cout << "Midpoint: " << midpoint.t() << std::endl;
	std::cout << "Bounding box lengths: " << bounding_box_lengths.t();
	#endif


	if (arma::norm(bounding_box_lengths) == 0) {
		#if KDTTREE_PC_DEBUG
		std::cout << "Cluttered node" << std::endl;
		#endif

		node -> set_is_cluttered(true);
		

		return node;
	}

	int longest_axis = bounding_box_lengths.index_max();

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
	node -> left = build(left_points, depth + 1);
	node -> right = build(right_points, depth + 1);

	return node;

}

std::vector<std::shared_ptr<PointNormal> > * KDTreePC::get_points_normals() {
	return &this -> points_normals;
}

std::shared_ptr<PointNormal> KDTreePC::get_point_normal(const int & i) const{
	return this -> points_normals.at(i);
}

unsigned int KDTreePC::get_size() const {
	return this -> points_normals.size();
}


