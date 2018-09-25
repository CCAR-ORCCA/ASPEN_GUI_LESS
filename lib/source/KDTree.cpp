#include <KDTree.hpp>
#include <PointDescriptor.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>



#define KDTTREE_DEBUG_FLAG 0
#define KDTREE_BUILD_DEBUG 0

template <class T> KDTree<T>::KDTree(PointCloud<T> * owner) {
this -> owner = owner;
}

template <class T> void KDTree<T>::set_depth(int depth) {
this -> depth = depth;
}


template <class T> double KDTree<T>::get_value() const {
return this -> value;
}

template <class T> unsigned int KDTree<T>::get_axis() const {
return this -> axis;
}

template <class T> void KDTree<T>::set_value(double value) {
this -> value = value;
}
template <class T>  void KDTree<T>::set_axis(unsigned int axis) {
this -> axis = axis;
}


template <class T>  void KDTree<T>::set_is_cluttered(bool cluttered){
this -> cluttered = cluttered;
}

template <class T>  bool KDTree<T>::get_is_cluttered() const{
return this -> cluttered;
}

template <class T>  void KDTree<T>::closest_point_search(const arma::vec & test_point,
const std::shared_ptr<KDTree> & node,
int & best_guess_index,
double & distance) const {

	if (node -> indices.size() == 1 || node -> get_is_cluttered() ) {

		double new_distance = this -> distance(this -> owner -> get_point(node -> indices[0]),test_point);


		if (new_distance < distance) {
			distance = new_distance;
			best_guess_index = node -> indices[0];
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
					best_guess_index,
					distance);
			}

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess_index,
					distance);
			}

		}

		else {

			if (test_point(node -> get_axis()) + distance > node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> right,
					best_guess_index,
					distance);
			}

			if (test_point(node -> get_axis()) - distance <= node -> get_value()) {
				node -> closest_point_search(test_point,
					node -> left,
					best_guess_index,
					distance);
			}

		}

	}


}

template <class T>  void KDTree<T>::closest_N_point_search(const arma::vec & test_point,
const unsigned int & N_points,
const std::shared_ptr<KDTree> & node,
double & distance,
std::map<double,int > & closest_points) const{

	#if KDTTREE_DEBUG_FLAG
	std::cout << "#############################\n";
	std::cout << "Depth: " << this -> depth << std::endl; ;
	std::cout << "Points in node: " << node -> indices.size() << std::endl;
	std::cout << "Points found so far : " << closest_points.size() << std::endl;
	#endif

	// DEPRECATED
	if (node -> indices.size() == 1 || node -> get_is_cluttered()) {

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Leaf node\n";
		#endif


		double new_distance = this -> distance(this -> owner -> get_point(node -> indices[0]),test_point);
		

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Distance to query_point: " << new_distance << std::endl;
		#endif

		if (closest_points.size() < N_points){
			closest_points[new_distance] = node -> indices[0];
		}
		else{

			unsigned int size_before = closest_points.size(); // should always be equal to N_points

			closest_points[new_distance] = node -> indices[0];

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

	#if KDTTREE_DEBUG_FLAG
	std::cout << "#############################\n";
	std::cout << " Leaving "<<  std::endl; ;
	std::cout << "Points found : " << std::endl;
	for (auto it = closest_points.begin(); it != closest_points.end(); ++it){
		std::cout << it -> first << " : " << this -> owner -> get_point_coordinates(it -> second).t().t();
	}
	#endif


}


template <class T>  
void KDTree<T>::radius_point_search(const arma::vec & test_point,
	const std::shared_ptr<KDTree> & node,
	const double & distance,
	std::vector< int > & closest_points_indices) {



	double node_value = node -> get_value();
	double test_value = test_point(node -> get_axis());


	#if KDTTREE_DEBUG_FLAG
	std::cout << "#############################\n";
	std::cout << "Depth: " << node -> depth << std::endl; ;
	std::cout << "Points in node: " << node -> indices.size() << std::endl;
	std::cout << "Points found so far : " << closest_points_indices.size() << std::endl;
	#endif

	if (node -> indices.size() == 1 || node -> get_is_cluttered() ) {

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Leaf node\n";
		#endif

		double new_distance = this -> distance(this -> owner -> get_point(node -> indices[0]),test_point);
		

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Distance to query_point: " << new_distance << std::endl;
		#endif

		if (new_distance < distance ) {
			for (int i = 0; i < node -> indices.size(); ++i){
				closest_points_indices.push_back(node -> indices[i]);
			}
			#if KDTTREE_DEBUG_FLAG
			std::cout << "Found closest point " << node -> indices[0] << " with distance = " + std::to_string(distance)<< " \n" << std::endl;
			#endif
		}

	}

	else {

		#if KDTTREE_DEBUG_FLAG
		std::cout << "Fork node\n";
		#endif

		

		if (test_value <= node_value) {

			#if KDTTREE_DEBUG_FLAG
			std::cout << "Searching left first\n";
			#endif


			if (test_value - distance <= node_value) {
				node -> radius_point_search(test_point,
					node -> left,
					distance,
					closest_points_indices);
			}

			if (test_value + distance >= node_value) {
				node -> radius_point_search(test_point,
					node -> right,
					distance,
					closest_points_indices);
			}

		}

		else {

			#if KDTTREE_DEBUG_FLAG
			std::cout << "Searching right first\n";
			#endif
			if (test_value + distance >= node_value) {
				node -> radius_point_search(test_point,
					node -> right,
					distance,
					closest_points_indices);
			}

			if (test_value - distance <= node_value) {
				node -> radius_point_search(test_point,
					node -> left,
					distance,
					closest_points_indices);
			}

		}

	}

}

template <class T> 
void KDTree<T>::build(const std::vector< int > & indices, int depth) {

	this -> indices = indices;
	this -> left = nullptr;
	this -> right = nullptr;
	this -> set_depth(depth);

	#if KDTREE_BUILD_DEBUG
	std::cout << "Points in node: " << indices.size() <<  std::endl;
	#endif

	if (this -> indices.size() == 0) {
		#if KDTREE_BUILD_DEBUG
		std::cout << "Empty node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif
		return;
	}
	else if (this -> indices.size() == 1){
		#if KDTREE_BUILD_DEBUG
		std::cout << "Trivial node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif
		return;
	}

	else {

		this -> left = std::make_shared<KDTree<T>>( KDTree<T>(this -> owner) );
		this -> right = std::make_shared<KDTree<T>>( KDTree<T>(this -> owner) );

		this -> left -> indices = std::vector<int >();
		this -> right -> indices = std::vector<int >();

	}

	arma::vec midpoint = arma::zeros<arma::vec>(this -> owner -> get_point_coordinates(indices[0]).size());
	const arma::vec & start_point = this -> owner -> get_point_coordinates(indices[0]);

	arma::vec min_bounds = start_point;
	arma::vec max_bounds = start_point;

	// Could multithread here
	for (unsigned int i = 0; i < indices.size(); ++i) {

		arma::vec point = this -> owner -> get_point_coordinates(indices[i]);

		max_bounds = arma::max(max_bounds,point);
		min_bounds = arma::min(min_bounds,point);

		// The midpoint of all the facets is found
		midpoint += (point / indices.size());
	}



	arma::vec bounding_box_lengths = max_bounds - min_bounds;

	// Facets to be assigned to the left and right nodes
	std::vector < int > left_points;
	std::vector < int > right_points;

	#if KDTREE_BUILD_DEBUG
	std::cout << "Midpoint: " << midpoint.t() << std::endl;
	std::cout << "Bounding box lengths: " << bounding_box_lengths.t();
	#endif


	if (arma::norm(bounding_box_lengths) == 0) {
		#if KDTREE_BUILD_DEBUG
		std::cout << "Cluttered node" << std::endl;
		#endif

		this -> set_is_cluttered(true);

		return;
	}

	int longest_axis = bounding_box_lengths.index_max();

	for (unsigned int i = 0; i < indices.size() ; ++i) {

		if (midpoint(longest_axis) >= this -> owner -> get_point_coordinates(indices[i]).at(longest_axis)) {
			left_points.push_back(indices[i]);
		}

		else {
			right_points.push_back(indices[i]);
		}

	}

	this -> set_axis(longest_axis);
	this -> set_value(midpoint(longest_axis));

	// I guess this could be avoided
	if (left_points.size() == 0 && right_points.size() > 0) {
		left_points = right_points;
	}

	if (right_points.size() == 0 && left_points.size() > 0) {
		right_points = left_points;
	}


	// Recursion continues
	this -> left -> build(left_points, depth + 1);
	this -> right -> build(right_points, depth + 1);

}

template <class T> unsigned int KDTree<T>::size() const {
return this -> indices.size();
}


template <>
double KDTree<PointNormal>::distance(const PointNormal & point_in_pc,
	const arma::vec & point) const{

	return arma::norm(point_in_pc.get_point_coordinates() - point);

}



template <>
double KDTree<PointDescriptor>::distance(const PointDescriptor & point_in_pc,
	const arma::vec & point) const{

	return point_in_pc.distance_to_descriptor(point);

}



// Explicit instantiations
template class KDTree<PointNormal> ;
template class KDTree<PointDescriptor> ;




