#include "Ray.hpp"
#include "DebugFlags.hpp"
#include "ShapeModel.hpp"
#include "FrameGraph.hpp"
#include "Bezier.hpp"
#include "Lidar.hpp"
#include "Facet.hpp"
#include "Element.hpp"
#include "ControlPoint.hpp"

#define SINGLE_FACET_RAY_CASTING_DEBUG 0
#define INTERSECTION_INSIDE_DEBUG 0


Ray::Ray(unsigned int row_index, unsigned int col_index, Lidar * lidar) {

	this -> lidar = lidar;
	this -> row_index = row_index;
	this -> col_index = col_index;

	// the origin of the ray is computed in the lidar plane
	// the outward line-of-sight direction is set to the +x direction
	// The bottom left of the focal plane (as seen from an observer looking inside
	// the lidar) is formed by the fixel of coordinates (0,0);

	double y, z;
	double x = this -> lidar -> get_focal_length(); // positions the pixel in the focal plane 
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
	
	this -> direction = arma::normalise(origin);
	this -> origin = arma::zeros<arma::vec>(3);

	this -> origin_target_frame = arma::zeros<arma::vec>(3);
	this -> direction_target_frame = arma::zeros<arma::vec>(3);
}



Ray::Ray(arma::vec::fixed<3> origin, arma::vec::fixed<3> direction){
	this -> origin = origin;
	this -> direction = direction;

	this -> origin_target_frame = origin;
	this -> direction_target_frame = direction;
}


int Ray::get_hit_element() {
	return this -> hit_element;
}


void Ray::set_hit_element(int element) {
	this -> hit_element = element;
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

const arma::vec::fixed<3> & Ray::get_direction() const{
	return this -> direction;
}


const arma::vec::fixed<3> & Ray::get_origin() const{
	return this -> origin;
}


const arma::vec::fixed<3> & Ray::get_direction_target_frame() const{
	return this -> direction_target_frame;
}


const arma::vec::fixed<3> & Ray::get_origin_target_frame() const {
	return this -> origin_target_frame;
}

int Ray::get_guess() const{
	return this -> guess;
}
void Ray::set_guess (int guess){
	this -> guess = guess;
}

void Ray::reset(ShapeModel<ControlPoint> * shape_model) {

	this -> true_range = std::numeric_limits<double>::infinity();
	this -> hit_element = -1;
	
	FrameGraph * frame_graph = this -> get_lidar() -> get_frame_graph();

	this -> direction_target_frame = frame_graph -> convert(this -> get_direction(), "L", shape_model -> get_ref_frame_name(), true);
	this -> origin_target_frame = frame_graph -> convert(this -> get_origin(), "L", shape_model -> get_ref_frame_name(), false);

	this -> incidence_angle = std::numeric_limits<double>::infinity();


}

bool Ray::intersection_inside(const arma::vec::fixed<3> & H, const Facet & facet, double tol) {


	const std::vector<int> & control_points = facet.get_points();


	#if INTERSECTION_INSIDE_DEBUG
	std::cout << "\t\tFacet is comprised of points " << control_points[0] << ", " << control_points[1] << ", " << control_points[2] << std::endl;
	#endif


	const arma::vec::fixed<3> P0 = facet.get_point_coordinates(control_points[0]);
	#if INTERSECTION_INSIDE_DEBUG
	std::cout << "\t\t P0: " << P0.t();
	#endif
	const arma::vec::fixed<3> P1 = facet.get_point_coordinates(control_points[1]);
	#if INTERSECTION_INSIDE_DEBUG
	std::cout << "\t\t P1: " << P1.t();
	#endif
	const arma::vec::fixed<3> P2 = facet.get_point_coordinates(control_points[2]);
	#if INTERSECTION_INSIDE_DEBUG
	std::cout << "\t\t P2: " << P2.t() << std::endl;
	#endif



	double epsilon = (facet.get_area()
		- 0.5 * (
			arma::norm(arma::cross(H - P0, H - P1))
			+ arma::norm(arma::cross(H - P1, H - P2))
			+ arma::norm(arma::cross(H - P2, H - P0)) ));

	// If true, the intersection point is inside the surface
	return (std::abs(epsilon) < tol);

	

}


arma::vec::fixed<3> Ray::get_impact_point() const {

	if (this -> true_range < std::numeric_limits<double>::infinity()) {
		
		return (this -> direction * this -> true_range + this -> origin);
	}
	else {
		throw std::runtime_error("Invalid ray");
	}

}


arma::vec::fixed<3> Ray::get_impact_point_target_frame() const {

	if (this -> true_range < std::numeric_limits<double>::infinity()) {
		
		return (this -> direction_target_frame * this -> true_range + this -> origin_target_frame);
	}
	else {
		throw std::runtime_error("Invalid ray");
	}

}

int Ray::get_super_element() const{
	return this -> super_element;
}


bool Ray::single_facet_ray_casting(const Facet & facet,bool store,bool outside) {

	// The ray is parametrized as R = At + B where (A,B) are respectively
	// the direction and the origin of the ray. For an intersection to
	// be valid, t must be positive
	const arma::vec::fixed<3> & n = facet.get_normal_coordinates();
	const arma::vec::fixed<3> & p = facet.get_center();

	#if SINGLE_FACET_RAY_CASTING_DEBUG
	std::cout << "\t\tFacet normal: " << n.t();
	std::cout << "\t\tFacet center: " << p.t() << std::endl;	
	#endif

	double t = arma::dot(n, p - this -> origin_target_frame) / arma::dot(n, this -> direction_target_frame);


	// The normal is facing the opposite way
	if (arma::dot(n,this -> direction_target_frame) > 0 && outside){

	#if SINGLE_FACET_RAY_CASTING_DEBUG
		std::cout << "\t\tNormal is facing the wrong way. No impact\n";
	#endif

		return false;
	}

	// If the range is positive, this is further tested for
	// potential intersection with this facet
	if (t > 0) {
		arma::vec::fixed<3> H = this -> direction_target_frame * t + this -> origin_target_frame;

		#if SINGLE_FACET_RAY_CASTING_DEBUG
		std::cout << "\t\tChecking if intersection is inside facet\n";
		#endif

		if (this -> intersection_inside(H, facet)) {

			// Corresponds to range attenuation in 
			// Amzajerdian, F., Hines, G. D., Roback, V. E., Petway, L. B., Barnes, B. W., Paul, F., … Bulyshev, A. (2015). 
			// Advancing Lidar Sensors Technologies for Next Generation Landing Missions (pp. 1–11). https://doi.org/10.2514/6.2015-0329

			double b = 1800.; // max range
			double a = (900. - b) / (60.);

			double incidence_angle = 180. / arma::datum::pi * std::acos(std::abs(arma::dot(this -> direction_target_frame,n)));
			double max_range = a * incidence_angle  + b;



			if (store){
				if (this -> true_range > t ) {
					this -> true_range = t;
					this -> hit_element = facet.get_global_index();
					this -> incidence_angle = incidence_angle;
				}
			}

			else{
				if (this -> true_range < t){
					return false;
				}
				this -> super_element = facet.get_super_element();
				this -> KD_impact = H;
			}

			return true;
		}



	}

	return false;

}

arma::vec::fixed<3> Ray::get_KD_impact() const{
	return this -> KD_impact;
}


bool Ray::single_patch_ray_casting(const Bezier & patch,double & u,double & v,bool use_KD_impact) {

	const arma::vec::fixed<3> & S = this -> origin_target_frame;
	const arma::vec::fixed<3> & dir = this -> direction_target_frame;

	arma::mat::fixed<3,3> u_tilde = RBK::tilde(dir);

	// The ray caster iterates until a valid intersect is found
	unsigned int N_iter_max = 20;

	// The barycentric coordinates are initialized at a planar guess
	arma::mat::fixed<3,2> E;
	E.col(0) = patch.get_point_coordinates(patch.get_degree(),0) - patch.get_point_coordinates(0,0);
	E.col(1) = patch.get_point_coordinates(0,patch.get_degree()) - patch.get_point_coordinates(0,0);

	arma::vec::fixed<2> chi;
	if (use_KD_impact){
		chi = arma::solve(E.t() * E,E.t() * (this -> get_KD_impact() - patch.get_point_coordinates(0,0)));
	} 
	else{
		chi(0) = 1./3;
		chi(1) = 1./3;
	}


	arma::vec::fixed<2> dchi = arma::zeros<arma::vec>(2);

	arma::mat::fixed<3,2> H = arma::zeros<arma::mat>(3,2);
	arma::vec::fixed<3> Y = arma::zeros<arma::vec>(3);
	arma::vec::fixed<3> impact;

	for (unsigned int i = 0; i < N_iter_max; ++i){
		double u_t = chi(0);
		double v_t = chi(1);

		impact = patch.evaluate(u_t,v_t);
		double distance = arma::norm(u_tilde*(S - impact));

		#if RAY_DEBUG
		std::cout << "Iter: " << i << " Distance: " << distance << " (u,v): " << u_t << " " << v_t << std::endl;
		#endif

		if (distance < 1e-5){

			#if RAY_DEBUG
			std::cout << "Converged." << std::endl;
			#endif

			if (arma::dot(patch.get_normal_coordinates(u_t,v_t),dir) > 0){
				#if RAY_DEBUG
				std::cout << "Spurious normal. rejected" << std::endl;

				#endif 

				return false;
			}


			if (u_t + v_t > 1. || u_t < 0. || v_t < 0. || u_t > 1. || v_t > 1.){

				#if RAY_DEBUG
				std::cout << "Invalid edge hit" << std::endl;
				#endif 
				
				u = u_t;
				v = v_t;

				return false;
			}

			if (this -> true_range > arma::norm(S - impact)){

				double incidence_angle = 180. / arma::datum::pi * std::acos(std::abs(arma::dot(this -> direction_target_frame,patch.get_normal_coordinates(u_t,v_t))));

				this -> true_range = arma::norm(S - impact);
				u = u_t;
				v = v_t;
				this -> set_impact_coords(u,v);
				this -> hit_element = patch.get_global_index();
				this -> incidence_angle = incidence_angle;
				
				return true;
			}

			else{

				#if RAY_DEBUG
				std::cout << "Current true range: " << this -> true_range << std::endl;
				std::cout << "Range: " << arma::norm(S - impact) << std::endl;
				#endif 

				return false;
			}
		}
		
		H = u_tilde * patch.partial_bezier( u_t, v_t);
		Y = u_tilde * (S - patch.evaluate(u_t,v_t));

		dchi = arma::solve(H.t() * H,H.t() * Y);

		chi += dchi;
		

	}
	return false;

}









double Ray::get_incidence_angle() const{
	return this -> incidence_angle;
}


void Ray::set_impact_coords(const double & u,const double & v){
	this -> u = u;
	this -> v = v;
}

void Ray::get_impact_coords(double & u_t, double & v_t){
	u_t = this -> u;
	v_t = this -> v;
}
