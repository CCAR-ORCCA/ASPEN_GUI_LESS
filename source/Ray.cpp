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
	double x = - this -> lidar -> get_focal_length();

	double py = this -> lidar -> get_size_y() / this -> lidar -> get_row_count();
	double pz = this -> lidar -> get_size_z() / this -> lidar -> get_col_count();

	// Z coordinate
	double a_z;
	if (this -> lidar -> get_col_count() - 1 > 0) {
		a_z = (this -> lidar -> get_size_z() - pz) / (this -> lidar -> get_col_count() - 1) ;
	}
	else {
		a_z = 0;
	}
	double b_z = 0.5 * ( - this -> lidar -> get_size_z() + pz );
	z =  a_z * col_index + b_z;

	// Y coordinate
	double a_y;
	if (this -> lidar -> get_row_count() - 1 > 0) {
		a_y = (this -> lidar -> get_size_y() - py) / (this -> lidar -> get_row_count() - 1) ;
	}
	else {
		a_y = 0;
	}
	double b_y = 0.5 * ( - this -> lidar -> get_size_y() + py );
	y = a_y * row_index + b_y;

	// Origin and direction
	arma::vec origin = {x, z, y};
	this -> origin = std::make_shared<arma::vec>(origin);
	this -> direction = std::make_shared<arma::vec>(arma::normalise( - origin));

	this -> origin_target_frame = std::make_shared<arma::vec>(arma::zeros<arma::vec>(3));
	this -> direction_target_frame = std::make_shared<arma::vec>(arma::zeros<arma::vec>(3));


}

Facet * Ray::get_true_hit_facet() {
	return this -> true_hit_facet;
}

Facet * Ray::get_computed_hit_facet() {
	return this -> computed_hit_facet;
}

double Ray::get_true_range() const {
	return this -> true_range;
}

double Ray::get_computed_range() const {
	return this -> computed_range;
}

void Ray::set_true_range(double true_range) {
	this -> true_range = true_range;
}

void Ray::set_computed_range(double computed_range) {
	this -> computed_range = computed_range;
}



Lidar * Ray::get_lidar() {
	return this -> lidar;
}

double Ray::get_range_residual() const {
	return this -> range_residual;
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

bool Ray::brute_force_ray_casting(bool computed_mes) {


	bool hit = false;

	// Every facet of the shape model is searched for a potential intersect
	struct CompareRanges range_comp;

	#pragma omp parallel for reduction(minimum:range_comp) if (USE_OMP_RAY)
	for (unsigned int facet_index = 0;
	        facet_index < this -> lidar -> get_shape_model() -> get_NFacets();
	        ++facet_index) {


		Facet * facet = this -> lidar -> get_shape_model() -> get_facets() -> at(facet_index);

		// The ray is parametrized as R = At + B where (A,B) are respectively
		// the direction and the origin of the ray. For an intersection to
		// be valid, t must be positive
		arma::vec * n = facet -> get_facet_normal();
		arma::vec * p = facet -> get_facet_center();
		double t = arma::dot(*n, *p - *this -> origin_target_frame) / arma::dot(*n, *this -> direction_target_frame);

		// If the range is positive, this is further tested for
		// potential intersection with this facet
		if (t > 0) {

			arma::vec H =  *this -> direction_target_frame * t + (*this -> origin_target_frame);

			// If the intersect is indise the facet
			if (this -> intersection_inside(H, facet) == true) {
				double range = t;
				hit = true;

				// If the corresponding distance is less that what was already found,
				// this is an interesting intersection to retain
				if (range < range_comp . range || (range_comp.range > 1e10) == true) {
					range_comp . range = range;
					range_comp . hit_facet = facet;

				}

			}

		}

	}

	// The observed target is the a-priori
	if (computed_mes == true) {
		this -> computed_range = range_comp.range;
		this -> computed_hit_facet = range_comp.hit_facet;
		this -> range_residual = this -> true_range - this -> computed_range;
	}

	// The observed target is the true target
	else {
		this -> true_range = range_comp.range;
		this -> true_hit_facet = range_comp.hit_facet;
	}

	return hit;

}

void Ray::reset(bool computed_mes) {


	if (computed_mes == true) {
		this -> computed_range = std::numeric_limits<double>::infinity();
		this -> computed_hit_facet = nullptr;
	}
	else {
		this -> true_range = std::numeric_limits<double>::infinity();
		this -> true_hit_facet = nullptr;
	}

	FrameGraph * frame_graph = this -> get_lidar() -> get_frame_graph();
	(*this -> direction_target_frame) = frame_graph -> convert(*this -> get_direction(), "L", this -> get_lidar() -> get_shape_model() -> get_ref_frame_name(), true);
	(*this -> origin_target_frame) = frame_graph -> convert(*this -> get_origin(), "L", this -> get_lidar() -> get_shape_model() -> get_ref_frame_name(), false);


}

bool Ray::intersection_inside(arma::vec & H, Facet * facet, double tol) {

	arma::vec * P0 = facet -> get_vertices() -> at(0) -> get_coordinates() ;
	arma::vec * P1 = facet -> get_vertices() -> at(1) -> get_coordinates() ;
	arma::vec * P2 = facet -> get_vertices() -> at(2) -> get_coordinates() ;

	double epsilon = (facet -> get_area()
	                  - 0.5 * (
	                      arma::norm(arma::cross(H - *P0, H - *P1))
	                      + arma::norm(arma::cross(H - *P1, H - *P2))
	                      + arma::norm(arma::cross(H - *P2, H - *P0)) ));
	// If true, the intersection point is inside the surface
	if (std::abs(epsilon) < tol) {
		return true;
	}
	else {
		return false;
	}

}


arma::vec Ray::get_impact_point(bool computed_mes) const {

	if (computed_mes) {
		if (this -> computed_range < std::numeric_limits<double>::infinity()) {
			return (*this -> direction) * this -> computed_range + (*this -> origin);
		}
		else {
			throw "Invalid ray";
		}

	}
	else {

		if (this -> true_range < std::numeric_limits<double>::infinity()) {
			return (*this -> direction) * this -> true_range + (*this -> origin);
		}
		else {
			throw "Invalid ray";
		}

	}
}



bool Ray::single_facet_ray_casting(Facet * facet, bool computed_mes ) {


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

		if (this -> intersection_inside(H, facet) == true) {
			if (computed_mes == true) {
				if (this -> computed_range > t) {
					this -> computed_range = t;
					this -> computed_hit_facet = facet;
				}

			}
			else {
				if (this -> true_range > t) {

					this -> true_range = t;
					this -> true_hit_facet = facet;
				}

			}
			return true;
		}


	}

	return false;

}

