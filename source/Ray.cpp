#include "Ray.hpp"

Ray::Ray(unsigned int row_index, unsigned int col_index, Lidar * lidar) {

	this -> lidar = lidar;
	this -> row_index = row_index;
	this -> col_index = col_index;

	// the origin of the ray is computed in the lidar plane
	// the outward line-of-sight direction is set to the +x direction
	// The bottom left of the focal plane (as seen from an observer looking inside
	// the lidar) is formed by the fixel of coordinates (0,0);

	double y, z;
	double x = this -> lidar -> get_focal_length();

	double pz = this -> lidar -> get_size_z() / this -> lidar -> get_z_res();
	double py = this -> lidar -> get_size_y() / this -> lidar -> get_y_res();

	// Z coordinate
	double a_z;
	if (this -> lidar -> get_z_res() - 1 > 0) {
		a_z = (this -> lidar -> get_size_z() - pz) / (this -> lidar -> get_z_res() - 1) ;
	}
	else {
		a_z = 0;
	}
	double b_z = 0.5 * ( - this -> lidar -> get_size_z() + pz );
	z =  a_z * row_index + b_z;

	// Y coordinate
	double a_y;
	if (this -> lidar -> get_y_res() - 1 > 0) {
		a_y = (this -> lidar -> get_size_y() - py) / (this -> lidar -> get_y_res() - 1) ;
	}
	else {
		a_y = 0;
	}
	double b_y = 0.5 * ( - this -> lidar -> get_size_y() + py );
	y = a_y * col_index + b_y;

	// Origin and direction
	arma::vec origin = {x, z, y};
	this -> origin = std::make_shared<arma::vec>(origin);
	this -> direction = std::make_shared<arma::vec>(arma::normalise( origin));

	this -> origin_target_frame = std::make_shared<arma::vec>(arma::zeros<arma::vec>(3));
	this -> direction_target_frame = std::make_shared<arma::vec>(arma::zeros<arma::vec>(3));


}

Facet * Ray::get_true_hit_facet() {
	return this -> true_hit_facet;
}

double Ray::get_true_range() const {
	return this -> true_range;
}

void Ray::set_true_range(double true_range) {
	this -> true_range = true_range;
}



Lidar * Ray::get_lidar() {
	return this -> lidar;
}

arma::vec * Ray::get_direction() {
	return this -> direction.get();
}


arma::vec * Ray::get_origin() {
	return this -> origin.get();
}


arma::vec * Ray::get_direction_target_frame() {
	return this -> direction_target_frame.get();
}


arma::vec * Ray::get_origin_target_frame() {
	return this -> origin_target_frame.get();
}

void Ray::reset(ShapeModel * shape_model) {

	this -> true_range = std::numeric_limits<double>::infinity();
	this -> true_hit_facet = nullptr;
	
	FrameGraph * frame_graph = this -> get_lidar() -> get_frame_graph();

	(*this -> direction_target_frame) = frame_graph -> convert(*this -> get_direction(), "L", shape_model -> get_ref_frame_name(), true);
	(*this -> origin_target_frame) = frame_graph -> convert(*this -> get_origin(), "L", shape_model -> get_ref_frame_name(), false);

}

bool Ray::intersection_inside(arma::vec & H, Facet * facet, double tol) {

	arma::vec * P0 = facet -> get_control_points() -> at(0) -> get_coordinates() ;
	arma::vec * P1 = facet -> get_control_points() -> at(1) -> get_coordinates() ;
	arma::vec * P2 = facet -> get_control_points() -> at(2) -> get_coordinates() ;

	double epsilon = (facet -> get_area()
		- 0.5 * (
			arma::norm(arma::cross(H - *P0, H - *P1))
			+ arma::norm(arma::cross(H - *P1, H - *P2))
			+ arma::norm(arma::cross(H - *P2, H - *P0)) ));

	// If true, the intersection point is inside the surface
	if (std::abs(epsilon) < tol)
		return true;

	else
		return false;

}


arma::vec Ray::get_impact_point() const {

	if (this -> true_range < std::numeric_limits<double>::infinity()) {
		return (*this -> direction) * this -> true_range + (*this -> origin);
	}
	else {
		throw std::runtime_error("Invalid ray");
	}

	
}

bool Ray::single_facet_ray_casting(Facet * facet) {

	// The ray is parametrized as R = At + B where (A,B) are respectively
	// the direction and the origin of the ray. For an intersection to
	// be valid, t must be positive
	arma::vec * n = facet -> get_facet_normal();
	arma::vec * p = facet -> get_facet_center();

	double t = arma::dot(*n, *p - *this -> origin_target_frame) / arma::dot(*n, *this -> direction_target_frame);

	// If the range is positive, this is further tested for
	// potential intersection with this facet
	if (t > 0) {
		arma::vec H = *this -> direction_target_frame * t + *this -> origin_target_frame;


		if (this -> intersection_inside(H, facet)) {

			if (this -> true_range > t) {

				this -> true_range = t;
				this -> true_hit_facet = facet;
			}

			
			return true;
		}


	}

	return false;

}

