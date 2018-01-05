#include "ShapeModel.hpp"
#include <chrono>


ShapeModel::ShapeModel() {

}

ShapeModel::ShapeModel(std::string ref_frame_name,
	FrameGraph * frame_graph) {
	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
}


std::string ShapeModel::get_ref_frame_name() const {
	return this -> ref_frame_name;
}


arma::mat ShapeModel::get_inertia() const {
	return this -> inertia;

}


void ShapeModel::construct_kd_tree_control_points(bool verbose) {

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	this -> kdt_control_points = std::make_shared<KDTree_control_points>(KDTree_control_points());
	this -> kdt_control_points = this -> kdt_control_points -> build(this -> control_points, 0, verbose);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;


	std::cout << "\n Elapsed time during KDTree construction : " << elapsed_seconds.count() << "s\n\n";

}

void ShapeModel::set_ref_frame_name(std::string ref_frame_name) {

	this -> ref_frame_name = ref_frame_name;
}


std::vector<std::shared_ptr< ControlPoint> > * ShapeModel::get_control_points() {
	return &this -> control_points;
}


std::shared_ptr< ControlPoint>  ShapeModel::get_control_point(unsigned int i) const {
	return this -> control_points.at(i);
}


std::shared_ptr<KDTree_control_points> ShapeModel::get_KDTree_control_points() const {
	return this -> kdt_control_points;
}


unsigned int ShapeModel::get_NElements() const {
	return this -> elements . size();
}

unsigned int ShapeModel::get_NControlPoints() const {
	return this -> control_points . size();
}

void ShapeModel::initialize_index_table(){
	// The forward look up table is created
	for (auto iter = this -> control_points.begin(); iter != this -> control_points.end(); ++iter){
		this -> pointer_to_global_index[*iter] = this -> pointer_to_global_index.size();
		
	}
}

arma::vec ShapeModel::get_center() const{
	arma::vec center = {0,0,0};
	unsigned int N = this -> get_NControlPoints();

	for (auto point = this -> control_points.begin(); point != this -> control_points.end(); ++point){
		center += (*point) -> get_coordinates() / N;
	}
	return center;
}


void ShapeModel::transform(arma::vec & translation,arma::mat & rotation, arma::vec & stretch){

	arma::vec center = this -> get_center();
	arma::mat stretch_mat = arma::diagmat(stretch);


	for (auto point = this -> control_points.begin(); point != this -> control_points.end(); ++point){

		auto owning_elements = (*point) -> get_owning_elements();
		bool ignore = false;
		
		for (auto el = owning_elements.begin(); el != owning_elements.end(); ++el){
			if ((*el) -> get_info_mat_ptr() != nullptr){
				ignore = true;
				break;
			}
		}
		if (ignore){
			continue;
		}

		arma::vec new_coords = rotation * (stretch_mat * ((*point) -> get_coordinates() - center) ) + center + translation;


		(*point) -> set_coordinates(new_coords);


	}



}






unsigned int ShapeModel::get_control_point_index(std::shared_ptr<ControlPoint> point) const{
	return this -> pointer_to_global_index.at(point);
}


std::vector<std::shared_ptr<Element> > * ShapeModel::get_elements() {
	return &this -> elements;
}


void ShapeModel::add_element(std::shared_ptr<Element> el) {
	this -> elements.push_back(el);
}


void ShapeModel::add_control_point(std::shared_ptr<ControlPoint> vertex) {
	this -> control_points.push_back(vertex);
}

