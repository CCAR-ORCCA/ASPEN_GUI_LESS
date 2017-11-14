#include "Bezier.hpp"
#include <cassert>
Bezier::Bezier(std::vector<std::shared_ptr<ControlPoint > > control_points) : Element(control_points){

	double n = (-3 + std::sqrt(9 - 8 * (1 - control_points.size() )))/2;
	double intpart;

	if (modf(n,&intpart) == 0){
		this -> n = (unsigned int)(n);
	}
	else{
		throw(std::runtime_error("The control points cardinal does not correspond to a bezier control net"));
	}

	
	this -> construct_index_tables();




}

void Bezier::construct_index_tables(){
	
	this -> global_to_i.clear();
	this -> global_to_j.clear();

	unsigned int index = 0;

	for (unsigned int i = 0; i < this -> n + 1; ++i){

		for (unsigned int k = 0; k < this -> n + 1 - i; ++ k){

			this -> global_to_i[index] = i;
			this -> global_to_j[index] = this -> n - i - k;

			++index;

		}
	}
}








unsigned int Bezier::get_n() const{

	return this -> n;
}


std::set < Element * > Bezier::get_neighbors(bool all_neighbors) const{
	std::set < Element * > neighbors;
	return neighbors;
}


arma::vec Bezier::evaluate(const double u, const double v) const{

	arma::vec P = arma::zeros<arma::vec>(3);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){
		unsigned int i = this -> global_to_i.at(l);
		unsigned int j = this -> global_to_j.at(l);

		P += this -> bernstein(u,v,i,j) * this -> control_points[l] -> get_coordinates();
	}
	return P;
}



double Bezier::bernstein(
	const double u, 
	const double v,
	const unsigned int i,
	const unsigned int j) const{



	double coef =  boost::math::factorial<double>(n) / (
		boost::math::factorial<double>(i)
		* boost::math::factorial<double>(j) 
		* boost::math::factorial<double>(n - i - j));


	return coef * (std::pow(u,i) *  std::pow(v,j)
		* std::pow(1 - u - v,this -> n - i - j ));

}

void Bezier::elevate_n(){

}

void Bezier::compute_normal(){

}
void Bezier::compute_area(){

}
void Bezier::compute_center(){
	this -> center = this -> evaluate(1./3.,1./3.);
}