#include "Bezier.hpp"




Bezier::Bezier( std::vector<std::shared_ptr<ControlPoint > > vertices): Element(control_points){

	double order = (-3 + std::sqrt(9 - 8 * (1 - vertices.size() )))/2;
	double intpart;
	if (modf(order,&intpart) == 0){
		this -> order = (unsigned int)(order / 2);
	}
	else{
		throw(std::runtime_error("The control points size does not correspond to a bezier control net"));
	}
}


unsigned int Bezier::get_order() const{

	return this -> order;
}


std::set < Element * > Bezier::get_neighbors(bool all_neighbors) const{
	std::set < Element * > neighbors;
	return neighbors;
}





void Bezier::compute_normal(){

}
void Bezier::compute_area(){

}
void Bezier::compute_center(){

}