#include "Element.hpp"


Element::Element(std::vector<std::shared_ptr<ControlPoint > > control_points) {
	this -> control_points = control_points;



}



std::vector<std::shared_ptr<ControlPoint > >  * Element::get_control_points() {
	return (&this -> control_points);
}

