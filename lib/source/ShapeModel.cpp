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


void ShapeModel::set_ref_frame_name(std::string ref_frame_name) {

	this -> ref_frame_name = ref_frame_name;
}


std::vector<std::shared_ptr< ControlPoint> > * ShapeModel::get_control_points() {
	return &this -> control_points;
}


unsigned int ShapeModel::get_NElements() const {
	return this -> elements . size();
}

unsigned int ShapeModel::get_NControlPoints() const {
	return this -> control_points . size();
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

