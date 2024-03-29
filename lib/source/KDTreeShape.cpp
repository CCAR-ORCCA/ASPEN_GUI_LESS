#include "KDTreeShape.hpp"
#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"

#include "DebugFlags.hpp"
#define KDTREE_SHAPE_HIT_DEBUG 0
#define KDTREE_SHAPE_BUILD_DEBUG 0


KDTreeShape::KDTreeShape(ShapeModelTri<ControlPoint> * owning_shape) {
	this -> owning_shape = owning_shape;
}


void KDTreeShape::build(const std::vector<int > & elements,  int depth) {

	// Creating the node
	this -> elements = elements;
	this -> left = std::make_shared<KDTreeShape>(KDTreeShape(this -> owning_shape));;
	this -> right = std::make_shared<KDTreeShape>(KDTreeShape(this -> owning_shape));;
	this -> set_depth(depth);
	this -> bbox = BBox(this -> owning_shape);

	#if KDTREE_SHAPE_BUILD_DEBUG
	std::cout << "Building node at depth " << depth << std::endl;
	std::cout << "Elements in node: " << elements.size() << std::endl;
	if(depth == 0){
		std::cout << "Owning shape has " << this -> owning_shape -> get_NElements() << " elements and " << this -> owning_shape -> get_NControlPoints() << " control points\n";
	}
	#endif

	if (elements.size() == 0) {
		#if KDTREE_SHAPE_BUILD_DEBUG
		std::cout << "Empty node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif
		return ;
	}

	// If the node only contains one triangle,
	// there's no point in subdividing it more
	if (elements.size() == 1) {


		#if KDTREE_SHAPE_BUILD_DEBUG
		std::cout << "Trivial node at depth " << depth << std::endl;
		#endif

		this -> bbox . update(elements[0]);

		return ;

	}

	#if KDTREE_SHAPE_BUILD_DEBUG
	std::cout << "Updating bounding box with elements in node\n";
	#endif 

	this -> bbox.update(elements);


	arma::vec::fixed<3> midpoint = arma::zeros<arma::vec>(3);

	#if KDTREE_SHAPE_BUILD_DEBUG

	std::cout << "Evaluating node midpoint\n";
	#endif
	// Could multithread here
	for (unsigned int i = 0; i < elements.size(); ++i) {
		// The midpoint of all the elements is found
		midpoint += (this -> owning_shape -> get_element(elements[i]).get_center() / elements.size());
	}


	// Facets to be assigned to the left and right nodes
	std::vector < int > left_facets;
	std::vector < int > right_facets;

	unsigned int longest_axis = this -> bbox.get_longest_axis();

	#if KDTREE_SHAPE_BUILD_DEBUG
	std::cout << "Node midpoint: " << midpoint.t() << std::endl;

	std::cout << "Elements in node: " << elements.size() << std::endl;
	this -> bbox.print();
	std::cout << "Longest bbox axis : " << longest_axis << std::endl;
	#endif


	for (unsigned int i = 0; i < elements.size() ; ++i) {

		bool added_to_left = false;
		bool added_to_right = false;

		for (unsigned int v = 0; v < 3; ++v) {

			// The elements currently owned by the node are split
			// based on where their vertices lie

			const Facet & element = this -> owning_shape -> get_element(elements[i]);
			const std::vector<int> & control_points = element.get_points();
			double vertex_coordinates_longest_axis = this -> owning_shape -> get_point_coordinates(control_points[v])(longest_axis);
			
			if ( midpoint(longest_axis) >= vertex_coordinates_longest_axis && !added_to_left ) {
				left_facets.push_back(elements[i]);
				added_to_left = true;
			}

			else if (midpoint(longest_axis) <= vertex_coordinates_longest_axis && !added_to_right) {
				right_facets.push_back(elements[i]);
				added_to_right = true;
			}

		}

	}

	#if KDTREE_SHAPE_BUILD_DEBUG
	std::cout << "Number of facets stored in the left child node: " << left_facets.size() << std::endl;
	std::cout << "Number of facets stored in the right child node: " << right_facets.size() << std::endl;
	#endif

	// // I guess this could be avoided
	// if (left_facets.size() == 0 && right_facets.size() > 0) {
	// 	left_facets = right_facets;
	// }

	// if (right_facets.size() == 0 && left_facets.size() > 0) {
	// 	right_facets = left_facets;
	// }

	unsigned int matches = 0;

	for (unsigned int i = 0; i < left_facets.size(); ++i) {
		for (unsigned int j = 0; j < right_facets.size(); ++j) {
			if (left_facets[i] == right_facets[j]) {
				++matches;
				break;
			}
		}
	}

	#if KDTREE_SHAPE_BUILD_DEBUG
	std::cout << "Proportion of duplicate facets: " << (double)matches / left_facets.size() * 100 << " %" << std::endl;
	#endif


	// Subdivision stops if at least 50% of triangles are shared amongst the two leaves
	// or if this node has reached the maximum depth
	// specified in KDTreeShape.hpp (1000 by default)

	if ((double)matches / left_facets.size() >= 0.5 || depth > this -> max_depth){

		
	#if KDTREE_SHAPE_BUILD_DEBUG
		std::cout << "Stopping recursion" << std::endl;
		std::cout << "Leaf depth: " << this -> depth << std::endl;

		std::cout << "Leaf contains: " << this -> elements.size() << " elements " << std::endl;
		this -> bbox.print();
		std::string path = std::to_string(rand() ) + ".obj";
		this -> bbox.save_to_file(path);
	#endif
		return;

	}

	else{


		this -> left -> build(left_facets, depth + 1);
		this -> right -> build(right_facets, depth + 1);

	}

}


bool KDTreeShape::hit(const std::shared_ptr<KDTreeShape> & node,
	Ray * ray,
	bool outside,
	ShapeModelBezier<ControlPoint> * shape_model_bezier) const {
	


	#if KDTREE_SHAPE_HIT_DEBUG
	std::cout << "Node depth: " << node -> depth << std::endl;
	#endif


	// Taking care of empty nodes here
	if (node -> elements.size() == 0){
		return false;
	}

	// Check if the ray intersects the bounding box of the given node

	if (node -> hit_bbox(ray)) {

		#if KDTREE_SHAPE_HIT_DEBUG
		std::cout << "The bounding box was hit\n";
		std::cout << "This node contains " << node -> elements.size() << " elements\n";
		std::cout << "The left child node contains " << node -> left -> elements.size() <<  " elements \n";
		std::cout << "The right child node contains " << node -> right -> elements.size() <<  " elements \n";
		#endif





		// If there are triangles in the child leaves, those are checked
		// for intersect. First, the method checks whether it is still on a branch
		if (node -> left -> elements.size() > 0 || node -> right -> elements.size() > 0) {

			bool hitleft = this -> hit(node -> left, ray,outside,shape_model_bezier);
			bool hitright = this -> hit(node -> right, ray,outside,shape_model_bezier);

			return (hitleft || hitright);

		}

		else {

			#if KDTREE_SHAPE_HIT_DEBUG
			std::cout << "Searching all triangles in node for intersect\n";
			#endif

			bool hit_element = false;

			// If not, the current node is a leaf
			// Note that all elements in the nodes must be searched
			// std::cout << "starting in node" << std::endl;
			for (unsigned int i = 0; i < node -> elements.size(); ++i) {


				#if KDTREE_SHAPE_HIT_DEBUG
				std::cout << "\tFacet " << i << " in node, global triangle index is " << node -> elements[i] << "\n";
				#endif

				// If there is a hit
				if (shape_model_bezier == nullptr){

					
					if (ray -> single_facet_ray_casting( this -> owning_shape -> get_element(node -> elements[i]),true,outside)){
						
						hit_element = true;
					}
				}

				// If the shape is a collection of Bezier patches, things get a bit more protracted
				else{


					// If the KD tree enclosing the shape is hit, there may be an impact over the bezier shape
					if (ray -> single_facet_ray_casting( this -> owning_shape -> get_element(node -> elements[i]),false)) {

						#if KDTREE_SHAPE_HIT_DEBUG
						std::cout << "\t Ray has hit enclosing triangle. Iterating on barycentric coordinates\n";
						#endif

						double u,v;
						u = 1e10;
						v = 1e10;


						const Bezier & patch = shape_model_bezier -> get_element(ray -> get_super_element());

						// If the bezier patch subtending the kd tree triangle is hit, then our
						// work here is done
						if (ray -> single_patch_ray_casting(patch,u,v)){
							hit_element = true;
						}

						// If no previous hit has been recorded for this ray, 
						// or if a closer hit may be found
						// the neighbors to the patch are searched

						if (!hit_element && (ray -> get_hit_element() == -1 
							|| arma::norm(ray -> get_KD_impact() - ray -> get_origin_target_frame()) <  ray -> get_true_range())) {

							if (std::abs(u) == 1e10){
								// If that is the case, there is no point in searching the neighbors.
								// the search did not converge at all
								continue;
							}

							// It would be nice to only search the neighbors that are susceptible to host 
							// the impact point. 
							

							std::set<int> neighbors = patch.get_neighbors( u,  v);
							neighbors.erase(patch.get_global_index());

							for (auto it = neighbors.begin(); it !=  neighbors.end(); ++it){
								
								const Bezier & n_patch = shape_model_bezier -> get_element(*it);
								
								if (ray -> single_patch_ray_casting(n_patch,u,v)){
									hit_element = true;
									break;
								}

							}


						}


					}
					else{
						#if KDTREE_SHAPE_HIT_DEBUG
						std::cout << "\t Ray has not hit enclosing triangle.\n";
						#endif
					}
				}

			}
			return hit_element;

		}
	}

	return false;

}

void KDTreeShape::set_depth(int depth) {
	this -> depth = depth;
}



bool KDTreeShape::hit_bbox(Ray * ray) const {

	const arma::vec::fixed<3> & u = ray -> get_direction_target_frame();
	const arma::vec::fixed<3> & origin = ray -> get_origin_target_frame();

	arma::vec all_t(6);

	all_t(0) = (this ->bbox . get_xmin() - origin(0)) / u(0);
	all_t(1) = (this ->bbox . get_xmax() - origin(0)) / u(0);

	all_t(2) = (this ->bbox . get_ymin() - origin(1)) / u(1);
	all_t(3) = (this ->bbox . get_ymax() - origin(1)) / u(1);

	all_t(4) = (this ->bbox . get_zmin() - origin(2)) / u(2);
	all_t(5) = (this ->bbox . get_zmax() - origin(2)) / u(2);

	arma::vec all_t_sorted = arma::sort(all_t);

	double t_test = 0.5 * (all_t_sorted(2) + all_t_sorted(3));

	#if KDTREE_SHAPE_HIT_DEBUG
	std::cout << "t_test = " << t_test << std::endl;
	#endif

	arma::vec::fixed<3> test_point = origin + t_test * (u);

	// If the current minimum range for this Ray is less than the distance to this bounding box,
	// this bounding box is ignored

	if (ray -> get_true_range() < all_t_sorted(2)) {
		return false;
	}

	if (test_point(0) <= this -> bbox . get_xmax() && test_point(0) >= this -> bbox . get_xmin()) {

		#if KDTREE_SHAPE_HIT_DEBUG
		std::cout << "\t First test passed\n";
		#endif



		if (test_point(1) <= this -> bbox . get_ymax() && test_point(1) >= this -> bbox.get_ymin()) {

			#if KDTREE_SHAPE_HIT_DEBUG
			std::cout << "\t\t Second test passed\n";
			#endif

			if (test_point(2) <= this -> bbox . get_zmax() && test_point(2) >= this -> bbox . get_zmin()) {

				#if KDTREE_SHAPE_HIT_DEBUG
				std::cout << "\t\t\t Third test passed, bbox was hit\n";
				#endif

				return true;

			}
		}
	}

	return false;


}





