#include "Element.hpp"



Element::Element(std::vector<std::shared_ptr<ControlPoint > > control_points) {
	this -> control_points = control_points;

	for (unsigned int vertex_index = 0; vertex_index < this -> control_points . size(); ++vertex_index) {
		this -> control_points[vertex_index]-> add_ownership(this);
	}


}



std::vector<std::shared_ptr<ControlPoint > >  * Element::get_control_points() {
	return (&this -> control_points);
}

