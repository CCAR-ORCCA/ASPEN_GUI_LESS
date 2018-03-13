#include "KDTreeShape.hpp"
#include "ShapeModelBezier.hpp"
#include "DebugFlags.hpp"

KDTreeShape::KDTreeShape() {

}


std::shared_ptr<KDTreeShape> KDTreeShape::build(std::vector<std::shared_ptr<Element > > & elements,  int depth) {

	// Creating the node
	std::shared_ptr<KDTreeShape> node = std::make_shared<KDTreeShape>( KDTreeShape() );
	node -> elements = elements;
	node -> left = nullptr;
	node -> right = nullptr;
	node -> set_depth(depth);

	node -> bbox = BBox();

	if (elements.size() == 0) {
		#if KDTTREE_SHAPE_DEBUG
		std::cout << "Empty node" << std::endl;
		std::cout << "Leaf depth: " << depth << std::endl;
		#endif
		return node;
	}

	// If the node only contains one triangle,
	// there's no point in subdividing it more
	if (elements.size() == 1) {

		node -> bbox . update(elements[0]);

		node -> left = std::make_shared<KDTreeShape>( KDTreeShape() );
		node -> right = std::make_shared<KDTreeShape>( KDTreeShape() );

		node -> left -> elements = std::vector<std::shared_ptr<Element> >();
		node -> right -> elements = std::vector<std::shared_ptr<Element> >();


		return node;

	}

	node -> bbox.update(elements);


	arma::vec midpoint = arma::zeros<arma::vec>(3);

	// Could multithread here
	for (unsigned int i = 0; i < elements.size(); ++i) {

		// The midpoint of all the elements is found


		midpoint += (elements[i] -> get_center() / elements.size());
	}

	// Facets to be assigned to the left and right nodes
	std::vector < std::shared_ptr<Element> > left_facets;
	std::vector < std::shared_ptr<Element> > right_facets;

	unsigned int longest_axis = node -> bbox.get_longest_axis();

	for (unsigned int i = 0; i < elements.size() ; ++i) {

		bool added_to_left = false;
		bool added_to_right = false;

		for (unsigned int v = 0; v < 3; ++v) {

			// The elements currently owned by the node are split
			// based on where their vertices lie

			if ( midpoint(longest_axis) >= elements[i] -> get_control_points() -> at(v) -> get_coordinates()(longest_axis)
				&& added_to_left == false) {
				left_facets.push_back(elements[i]);
			added_to_left = true;
		}

		else if (midpoint(longest_axis) <= elements[i] -> get_control_points() -> at(v) -> get_coordinates()(longest_axis)
			&& added_to_right == false) {
			right_facets.push_back(elements[i]);
		added_to_right = true;
	}

}

}

	// I guess this could be avoided
if (left_facets.size() == 0 && right_facets.size() > 0) {
	left_facets = right_facets;
}

if (right_facets.size() == 0 && left_facets.size() > 0) {
	right_facets = left_facets;
}

unsigned int matches = 0;

for (unsigned int i = 0; i < left_facets.size(); ++i) {
	for (unsigned int j = 0; j < right_facets.size(); ++j) {
		if (left_facets[i] == right_facets[j]) {
			++matches;
		}
	}
}



	// Subdivision stops if at least 50% of triangles are shared amongst the two leaves
	// or if this node has reached the maximum depth
	// specified in KDTreeShape.hpp (1000 by default)
if ((double)matches / left_facets.size() < 0.5 && (double)matches / right_facets.size() < 0.5 && depth < this -> max_depth) {


		// Recursion continues
	node -> left = build(left_facets, depth + 1);
	node -> right = build(right_facets, depth + 1);

}

else {

	node -> left = std::make_shared<KDTreeShape>( KDTreeShape() );
	node -> right = std::make_shared<KDTreeShape>( KDTreeShape() );

	node -> left -> elements = std::vector<std::shared_ptr<Element> >();
	node -> right -> elements = std::vector<std::shared_ptr<Element> >();

	#if KDTTREE_SHAPE_DEBUG

	std::cout << "Leaf depth: " << depth << std::endl;
	std::cout << "Leaf contains: " << node -> elements.size() << " elements " << std::endl;

	node -> bbox.print();
	std::string path = std::to_string(rand() ) + ".obj";
	node -> bbox.save_to_file(path);

	#endif

}

return node;

}


bool KDTreeShape::hit(KDTreeShape * node,
	Ray * ray,
	ShapeModelBezier * shape_model_bezier) const {
	

	// Check if the ray intersects the bounding box of the given node

	if (node -> hit_bbox(ray)) {

		// If there are triangles in the child leaves, those are checked
		// for intersect. First, the method checks whether it is still on a branch
		if (node -> left -> elements.size() > 0 || node -> right -> elements.size() > 0) {

			bool hitleft = this -> hit(node -> left.get(), ray,shape_model_bezier);
			bool hitright = this -> hit(node -> right.get(), ray,shape_model_bezier);

			return (hitleft || hitright);

		}

		else {
			
			bool hit_element = false;

			// If not, the current node is a leaf
			// Note that all elements in the nodes must be searched
			// std::cout << "starting in node" << std::endl;
			for (unsigned int i = 0; i < node -> elements.size(); ++i) {

				// If there is a hit
				if (shape_model_bezier == nullptr){
					if (ray -> single_facet_ray_casting( static_cast<Facet * >(node -> elements[i].get()))){
						hit_element = true;
					}
				}
				// If the shape is a collection of Bezier patches, things get a bit more protracted
				else{
					// If the KD tree enclosing the shape is hit, there may be an impact over the bezier shape
					if (ray -> single_facet_ray_casting( static_cast<Facet * >(node -> elements[i].get()),false)) {

						double u,v;
						u = 1e10;
						v = 1e10;


						Bezier * patch = static_cast<Bezier *>(ray -> get_super_element());

						// If the bezier patch subtending the kd tree triangle is hit, then our
						// work here is done
						if (ray -> single_patch_ray_casting(patch,u,v)){
							hit_element = true;
						}

						// If no previous hit has been recorded for this ray, 
						// or if a closer hit may be found
						// the neighbors to the patch are searched

						if (!hit_element && (ray -> get_hit_element() == nullptr 
							|| arma::norm(ray -> get_KD_impact() - *ray -> get_origin_target_frame()) <  ray -> get_true_range())) {

							if (std::abs(u) == 1e10){
								// If that is the case, there is no point in searching the neighbors.
								// the search did not converge at all
								continue;
							}

							// It would be nice to only search the neighbors that are susceptible to host 
							// the impact point. 
							

							auto neighbors = patch -> get_neighbors( u,  v);
							neighbors.erase(patch);

							for (auto it = neighbors.begin(); it !=  neighbors.end(); ++it){
								
								Bezier * n_patch = static_cast<Bezier *>(*it);
								if (ray -> single_patch_ray_casting(n_patch,u,v)){
									hit_element = true;
									break;
								}

							}


						}


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

	arma::vec * u = ray -> get_direction_target_frame();
	arma::vec * origin = ray -> get_origin_target_frame();


	arma::vec all_t(6);

	all_t(0) = (this ->bbox . get_xmin() - origin -> at(0)) / u -> at(0);
	all_t(1) = (this ->bbox . get_xmax() - origin -> at(0)) / u -> at(0);

	all_t(2) = (this ->bbox . get_ymin() - origin -> at(1)) / u -> at(1);
	all_t(3) = (this ->bbox . get_ymax() - origin -> at(1)) / u -> at(1);

	all_t(4) = (this ->bbox . get_zmin() - origin -> at(2)) / u -> at(2);
	all_t(5) = (this ->bbox . get_zmax() - origin -> at(2)) / u -> at(2);

	arma::vec all_t_sorted = arma::sort(all_t);

	double t_test = 0.5 * (all_t_sorted(2) + all_t_sorted(3));

	arma::vec test_point = *origin + t_test * (*u);

	// If the current minimum range for this Ray is less than the distance to this bounding box,
	// this bounding box is ignored

	if (ray -> get_true_range() < all_t_sorted(2)) {
		return false;
	}


	if (test_point(0) <= this -> bbox . get_xmax() && test_point(0) >= this -> bbox . get_xmin()) {

		if (test_point(1) <= this -> bbox . get_ymax() && test_point(1) >= this -> bbox  .get_ymin()) {

			if (test_point(2) <= this -> bbox . get_zmax() && test_point(2) >= this -> bbox . get_zmin()) {


				return true;

			}
		}
	}

	return false;


}





