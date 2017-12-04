#include "ShapeModelBezier.hpp"

ShapeModelBezier::ShapeModelBezier(ShapeModelTri * shape_model,
	std::string ref_frame_name,
	FrameGraph * frame_graph): ShapeModel(ref_frame_name,frame_graph){

	// All the facets of the original shape model are browsed
	// The shape starts as a uniform union of order-2 Bezier patches

	// The control point of this shape model are the same as that
	// of the provided shape
	this -> control_points = (*shape_model -> get_control_points());

	// The ownership relationships are reset
	for (unsigned int i = 0; i < shape_model -> get_NControlPoints(); ++i){
		this -> control_points[i] -> reset_ownership();
	}

	// The surface elements are almost the same, expect that they are 
	// Bezier patches and not facets
	for (unsigned int i = 0; i < shape_model -> get_NElements(); ++i){
		auto patch = std::make_shared<Bezier>(Bezier(*shape_model -> get_elements() -> at(i) -> get_control_points()));

		this -> elements.push_back(patch);

		patch -> get_control_points() -> at(0) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(1) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(2) -> add_ownership(patch.get());

	}

	this -> construct_kd_tree_control_points();

}

std::shared_ptr<arma::sp_mat> ShapeModelBezier::get_info_mat_ptr() const{
	return this -> info_mat_ptr;
}


void ShapeModelBezier::initialize_info_mat(){
	unsigned int N = this -> control_points.size();
	this -> info_mat_ptr = std::make_shared<arma::sp_mat>(arma::sp_mat(3 * N,3 * N ));
}

ShapeModelBezier::ShapeModelBezier(std::string ref_frame_name,
	FrameGraph * frame_graph): ShapeModel(ref_frame_name,frame_graph){

}

ShapeModelBezier::ShapeModelBezier(Bezier patch){

	this -> elements.push_back(std::make_shared<Bezier>(patch));
}

void ShapeModelBezier::compute_surface_area(){

}

void ShapeModelBezier::compute_volume(){

}

void ShapeModelBezier::compute_center_of_mass(){

}

void ShapeModelBezier::compute_inertia(){

}

bool ShapeModelBezier::ray_trace(Ray * ray){

	return true;

}

void ShapeModelBezier::elevate_degree(){

	// All patches are elevated
	for (unsigned int i = 0; i < this -> get_NElements(); ++i){
		dynamic_cast<Bezier *>(this -> get_elements() -> at(i).get()) -> elevate_degree();
	}

	std::vector<std::shared_ptr<ControlPoint> > new_control_points;
	std::set<std::shared_ptr<ControlPoint> > new_control_points_set;



	for (unsigned int i = 0; i < this -> get_NElements(); ++i){

		auto points = this -> get_elements() -> at(i) -> get_control_points();
		
		for (auto point = points -> begin(); point != points -> end(); ++point){
			if (new_control_points_set.find(*point) == new_control_points_set.end()){
				new_control_points_set.insert(*point) ;
				new_control_points.push_back(*point);
			}
		}
	}


	// The control point of this shape model are the same as that
	// of the provided shape
	this -> control_points = new_control_points;

	// The ownership relationships are reset
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> reset_ownership();
	}

	// The surface elements are almost the same, expect that they are 
	// Bezier patches and not facets
	for (auto patch = this -> elements.begin(); patch != this -> elements.end(); ++patch){

		auto points = (*patch) -> get_control_points();

		for (auto point = points -> begin(); point != points -> end(); ++point){

			(*point) -> add_ownership(patch -> get());

		}

	
	}


	this -> construct_kd_tree_control_points();
	
}

void ShapeModelBezier::save(std::string path) {
	// An inverse map going from vertex pointer to global indices is created

	std::map<std::shared_ptr<ControlPoint> , unsigned int> pointer_to_global_indices;
	std::map<unsigned int,std::shared_ptr<ControlPoint> > global_index_to_pointer;

	std::vector<arma::vec> vertices;
	std::vector< std::vector<unsigned int> > shape_patch_indices;

	// The global indices of the control points are found. 
	for (unsigned int i = 0; i < this -> get_NElements(); ++i){

		auto patch = this -> get_elements() -> at(i);

		std::vector<unsigned int> patch_indices;

		for (unsigned int index = 0; index < patch -> get_control_points() -> size(); ++index){

			if (pointer_to_global_indices.find(patch -> get_control_points() -> at(index)) == pointer_to_global_indices.end()){
				pointer_to_global_indices[patch -> get_control_points() -> at(index)] = pointer_to_global_indices.size();
				global_index_to_pointer[pointer_to_global_indices.size()] = patch -> get_control_points() -> at(index);
			}

			patch_indices.push_back(pointer_to_global_indices[patch -> get_control_points() -> at(index)]);

		}



		shape_patch_indices.push_back(patch_indices);

	}

	// The coordinates are written to a file
	std::ofstream shape_file;
	shape_file.open(path);
	shape_file << this -> get_degree() << "\n";

	for (auto iter = global_index_to_pointer.begin(); iter != global_index_to_pointer.end(); ++iter){
		shape_file << "v " << iter -> second -> get_coordinates()(0) << " " << iter -> second -> get_coordinates()(1) << " " << iter -> second -> get_coordinates()(2) << "\n";
	}

	for (auto iter = shape_patch_indices.begin(); iter != shape_patch_indices.end(); ++iter){
		shape_file << "f ";
		
		for (unsigned int index = 0; index < iter -> size(); ++index){
			
			if (index != iter -> size() - 1){
				shape_file << iter -> at(index) << " ";
			}
			else if (index == iter -> size() - 1 && iter != shape_patch_indices.end() - 1 ){
				shape_file << iter -> at(index) << "\n";
			}
			else{
				shape_file << iter -> at(index);
			}
		}

	}

}

unsigned int ShapeModelBezier::get_degree(){
	if (this -> get_elements() -> size() == 0){
		throw(std::runtime_error("This bezier shape model has no elements"));
	}
	return dynamic_cast<Bezier * >(this -> elements. begin() -> get()) -> get_degree();
}


void ShapeModelBezier::save_to_obj(std::string path) {

	// An inverse map going from vertex pointer to global indices is created

	// Note that the actual vertices on the shape model will not be be 
	// the control points, but the points lying on the bezier patch
 	// they support

	std::map<std::shared_ptr<ControlPoint> , unsigned int> pointer_to_global_indices;
	std::vector<arma::vec> vertices;
	std::vector<std::tuple<std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint> > > facets;


	





	// The global indices of the control points are found. 
	for (unsigned int i = 0; i < this -> get_NElements(); ++i){

		Bezier * patch = dynamic_cast<Bezier * >(this -> get_elements() -> at(i).get());

		for (unsigned int index = 0; index < patch -> get_control_points() -> size(); ++index){

			if (pointer_to_global_indices.find(patch -> get_control_points() -> at(index))== pointer_to_global_indices.end()){
				pointer_to_global_indices[patch -> get_control_points() -> at(index)] = pointer_to_global_indices.size();
				
				auto local_indices = patch -> get_local_indices(patch -> get_control_points() -> at(index));
				double u =  double(std::get<0>(local_indices)) / patch -> get_degree();
				double v =  double(std::get<1>(local_indices)) / patch -> get_degree();

				arma::vec surface_point = patch -> evaluate(u,v);
				vertices.push_back(surface_point);
			}

		}


	// The facets are created

		for (unsigned int l = 0; l < patch -> get_degree(); ++l){

			for (unsigned int t = 0; t < l + 1; ++t){

				if (t <= l){

					std::shared_ptr<ControlPoint> v0 = patch -> get_control_point(patch -> get_degree() - l,l - t);
					std::shared_ptr<ControlPoint> v1 = patch -> get_control_point(patch -> get_degree() - l - 1,l - t + 1);
					std::shared_ptr<ControlPoint> v2 = patch -> get_control_point(patch -> get_degree() - l - 1,l-t);

					facets.push_back(std::make_tuple(v0,v1,v2));
				}

				if (t > 0 ){

					std::shared_ptr<ControlPoint> v0 = patch -> get_control_point(patch -> get_degree() - l,l-t);
					std::shared_ptr<ControlPoint> v1 = patch -> get_control_point(patch -> get_degree() - l,l - t + 1 );
					std::shared_ptr<ControlPoint> v2 = patch -> get_control_point(patch -> get_degree() - l -1,l - t + 1);

					facets.push_back(std::make_tuple(v0,v1,v2));
				}

			}

		}
	}

	// The coordinates are written to a file

	std::ofstream shape_file;
	shape_file.open(path);

	for (unsigned int i = 0; i < vertices.size(); ++i){
		shape_file << "v " << vertices[i](0) << " " << vertices[i](1) << " " << vertices[i](2) << "\n";
	}

	for (unsigned int i = 0; i < facets.size(); ++i){
		unsigned int indices[3];
		indices[0] = pointer_to_global_indices[std::get<0>(facets[i])] + 1;
		indices[1] = pointer_to_global_indices[std::get<1>(facets[i])] + 1;
		indices[2] = pointer_to_global_indices[std::get<2>(facets[i])] + 1;



		shape_file << "f " << indices[0] << " " << indices[1] << " " << indices[2] << "\n";

	}


}






