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

