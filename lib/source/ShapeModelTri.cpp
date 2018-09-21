#include "ShapeModelTri.hpp"
#include "Facet.hpp"
#include "Ray.hpp"
#include <boost/progress.hpp>


#pragma omp declare reduction (+ : arma::vec::fixed<3> : omp_out += omp_in) \
initializer( omp_priv = arma::zeros<arma::vec>(3) )

void ShapeModelTri::update_mass_properties() {



	std::cout << "Computing mass properties...\n";
	std::cout << "\t Computing surface_area...\n";

	this -> compute_surface_area();

	std::cout << "\t Computing volume...\n";

	this -> compute_volume();
	
	std::cout << "\t Computing center of mass...\n";
	
	this -> compute_center_of_mass();
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	
	std::cout << "\t Computing inertia...\n";

	this -> compute_inertia();
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time in ShapeModelTri::update_mass_properties: " << elapsed_seconds.count() << " s"<< std::endl;
	

}

void ShapeModelTri::update_facets() {

	for (auto & facet : this -> elements) {
		facet -> update();
	}

}

void ShapeModelTri::update_facets(std::set<Facet *> & elements) {

	for (auto & facet : elements) {
		facet -> update();
	}

}


bool ShapeModelTri::ray_trace(Ray * ray){

	return this -> kdt_facet -> hit(this -> get_KDTreeShape().get(),ray);
}



void ShapeModelTri::construct_kd_tree_shape() {

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	this -> kdt_facet = std::make_shared<KDTreeShape>(KDTreeShape());
	this -> kdt_facet = this -> kdt_facet -> build(this -> elements, 0);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	std::cout << "\n Elapsed time during polyhedron KDTree construction : " << elapsed_seconds.count() << "s\n\n";

}







bool ShapeModelTri::contains(double * point, double tol ) {

	double lagrangian = 0;

	// Facet loop
	#// pragma omp parallel for reduction(+:lagrangian) if (USE_OMP_DYNAMIC_ANALYSIS)
	for (unsigned int facet_index = 0; facet_index < this -> get_NElements(); ++ facet_index) {

		std::vector<std::shared_ptr<ControlPoint > > * vertices = this -> get_elements() -> at(facet_index) -> get_control_points();

		const double * r1 =  vertices -> at(0) -> get_coordinates().colptr(0);
		const double * r2 =  vertices -> at(1) -> get_coordinates().colptr(0);
		const double * r3 =  vertices -> at(2) -> get_coordinates().colptr(0);

		double r1m[3];
		double r2m[3];
		double r3m[3];

		r1m[0] = r1[0] - point[0];
		r1m[1] = r1[1] - point[1];
		r1m[2] = r1[2] - point[2];

		r2m[0] = r2[0] - point[0];
		r2m[1] = r2[1] - point[1];
		r2m[2] = r2[2] - point[2];

		r3m[0] = r3[0] - point[0];
		r3m[1] = r3[1] - point[1];
		r3m[2] = r3[2] - point[2];


		double R1 = std::sqrt( r1m[0] * r1m[0]
			+ r1m[1] * r1m[1]
			+ r1m[2] * r1m[2]       );

		double R2 = std::sqrt( r2m[0] * r2m[0]
			+ r2m[1] * r2m[1]
			+ r2m[2] * r2m[2]      );


		double R3 = std::sqrt( r3m[0] * r3m[0]
			+ r3m[1] * r3m[1]
			+ r3m[2] * r3m[2]      );

		double r2_cross_r3_0 = r2m[1] * r3m[2] - r2m[2] * r3m[1];
		double r2_cross_r3_1 = r3m[0] * r2m[2] - r3m[2] * r2m[0];
		double r2_cross_r3_2 = r2m[0] * r3m[1] - r2m[1] * r3m[0];


		double wf = 2 * std::atan2(
			r1m[0] * r2_cross_r3_0 + r1m[1] * r2_cross_r3_1 + r1m[2] * r2_cross_r3_2,

			R1 * R2 * R3 + R1 * (r2m[0] * r3m[0] + r2m[1] * r3m[1]  + r2m[2] * r3m[2] )
			+ R2 * (r3m[0] * r1m[0] + r3m[1] * r1m[1] + r3m[2] * r1m[2])
			+ R3 * (r1m[0] * r2m[0] + r1m[1] * r2m[1] + r1m[2] * r2m[2]));



		lagrangian += wf;

	}

	if (std::abs(lagrangian) < tol) {
		return false;
	}
	else {
		return true;
	}

}


void ShapeModelTri::random_sampling(unsigned int N,arma::mat & points, arma::mat & normals) const{

	std::cout << " - Sampling surface points from the true shape model ...\n";

	int N_points_per_element = int(double(N) / this -> elements.size());

	points = arma::zeros<arma::mat>(3,N_points_per_element * this -> elements.size());
	normals = arma::zeros<arma::mat>(3, N_points_per_element * this -> elements.size());

	// N points are randomly sampled from the surface of the shape model
	boost::progress_display progress(this -> elements.size());
	
	// #pragma omp parallel for
	for (unsigned int f = 0; f < this -> elements.size(); ++f){

		auto vertices = this -> elements[f] -> get_control_points();

		auto V0 = vertices -> at(0) -> get_coordinates();
		auto V1 = vertices -> at(1) -> get_coordinates();
		auto V2 = vertices -> at(2) -> get_coordinates();

		arma::vec noise_intensity = 1e-1 * arma::randu<arma::vec>(1);

		for (int i = 0; i < N_points_per_element; ++i){

			arma::vec random = arma::randu<arma::vec>(2);
			double u = random(0);
			double v = random(1);

			points.col(N_points_per_element * f + i) = (1 - std::sqrt(u)) * V0 + std::sqrt(u) * ( 1 - v) * V1 + std::sqrt(u) * v * V2 +  noise_intensity(0) * arma::randn<arma::vec>(3);
			normals.col(N_points_per_element * f + i) = arma::normalise(arma::normalise(arma::cross(V1 - V0,V2 - V0)) + 0.1 * arma::randn<arma::vec>(3));

		}
		++progress;
	}

}








void ShapeModelTri::save(std::string path,const arma::vec & X,const arma::mat & M) const {
	std::ofstream shape_file;
	shape_file.open(path);

	

	std::map<std::shared_ptr<ControlPoint> , unsigned int> vertex_ptr_to_index;

	for (unsigned int vertex_index = 0;vertex_index < this -> get_NControlPoints();++vertex_index) {

		arma::vec coords = this -> control_points[vertex_index] -> get_coordinates();

		coords = M * coords + X;

		shape_file << "v " << coords(0)  << " " << coords(1) << " " << coords(2) << std::endl;
		vertex_ptr_to_index[this -> control_points[vertex_index]] = vertex_index;
	}

	for (unsigned int facet_index = 0;
		facet_index < this -> get_NElements();
		++facet_index) {

		unsigned int v0 =  vertex_ptr_to_index[this -> elements[facet_index] -> get_control_points() -> at(0)] + 1;
	unsigned int v1 =  vertex_ptr_to_index[this -> elements[facet_index] -> get_control_points() -> at(1)] + 1;
	unsigned int v2 =  vertex_ptr_to_index[this -> elements[facet_index] -> get_control_points() -> at(2)] + 1;

	shape_file << "f " << v0 << " " << v1 << " " << v2 << std::endl;

}


shape_file.close();




}










void ShapeModelTri::check_normals_consistency(double tol) const {
	double facet_area_average = 0;

	arma::vec surface_sum= arma::zeros<arma::vec>(3);

	for (unsigned int facet_index = 0; facet_index < this -> elements.size(); ++facet_index) {

		std::shared_ptr<Element> facet = this -> elements[facet_index];

		surface_sum += facet -> get_area() * facet -> get_normal_coordinates();
		facet_area_average += facet -> get_area();

	}


	facet_area_average = facet_area_average / this -> elements.size();
	if (arma::norm(surface_sum) / facet_area_average > tol) {
		std::cout <<  "Warning : normals were incorrectly oriented. norm(sum(n * s))/sum(s)= " + std::to_string(arma::norm(surface_sum) / facet_area_average) << std::endl;
	}

}



void ShapeModelTri::compute_volume() {
	double volume = 0;

	#pragma omp parallel for reduction(+:volume) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;
		facet_index < this -> elements.size();
		++facet_index) {

		std::vector<std::shared_ptr<ControlPoint > > * vertices = this -> elements[facet_index] -> get_control_points();

	arma::vec r0 =  vertices -> at(0) -> get_coordinates();
	arma::vec r1 =  vertices -> at(1) -> get_coordinates();
	arma::vec r2 =  vertices -> at(2) -> get_coordinates();
	double dv = arma::dot(r0, arma::cross(r1 - r0, r2 - r0)) / 6.;
	volume = volume + dv;

}

this -> volume = volume;

}




void ShapeModelTri::compute_center_of_mass() {
	
	double volume = this -> get_volume();
	double cx = 0;
	double cy = 0;
	double cz = 0;

	#pragma omp parallel for reduction (+:cx,cy,cz) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;facet_index < this -> elements.size();++facet_index) {

		std::vector<std::shared_ptr<ControlPoint > > * vertices = this -> elements[facet_index] -> get_control_points();

		arma::vec r0 =  vertices -> at(0) -> get_coordinates();
		arma::vec r1 =  vertices -> at(1) -> get_coordinates();
		arma::vec r2 =  vertices -> at(2) -> get_coordinates();

		
		double dv = 1. / 6. * arma::dot(r1, arma::cross(r1 - r0, r2 - r0));


		// C += (r0 + r1 + r2) / 4 * dv / volume;

		double coef =  dv / (4 * volume);
		cx = (r0(0) + r1(0) + r2(0)) * coef;
		cy = (r0(1) + r1(1) + r2(1)) * coef;
		cz = (r0(2) + r1(2) + r2(2)) * coef;


	}


	this -> cm =  {cx,cy,cz};


}


void ShapeModelTri::compute_inertia() {


	double P_xx = 0;
	double P_yy = 0;
	double P_zz = 0;
	double P_xy = 0;
	double P_xz = 0;
	double P_yz = 0;

	// double l = std::pow(this -> volume, 1. / 3.);
	double l = 1;


	# pragma omp parallel for reduction(+:P_xx,P_yy,P_zz,P_xy,P_xz,P_yz) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;
		facet_index < this -> elements.size();
		++facet_index) {


		std::vector<std::shared_ptr<ControlPoint > > * vertices = this -> elements[facet_index] -> get_control_points();

		// Normalized coordinates
	arma::vec r0 =  (vertices -> at(0) -> get_coordinates()) / l;
	arma::vec r1 =  (vertices -> at(1) -> get_coordinates()) / l;
	arma::vec r2 =  (vertices -> at(2) -> get_coordinates()) / l;

	double * r0d =  r0. colptr(0);
	double * r1d =  r1. colptr(0);
	double * r2d =  r2. colptr(0);

	double dv = 1. / 6. * arma::dot(r1, arma::cross(r1 - r0, r2 - r0));



	P_xx += dv / 20 * (2 * r0d[0] * r0d[0]
		+ 2 * r1d[0] * r1d[0]
		+ 2 * r2d[0] * r2d[0]
		+ r0d[0] * r1d[0]
		+ r0d[0] * r1d[0]
		+ r0d[0] * r2d[0]
		+ r0d[0] * r2d[0]
		+ r1d[0] * r2d[0]
		+ r1d[0] * r2d[0]);


	P_yy += dv / 20 * (2 * r0d[1] * r0d[1]
		+ 2 * r1d[1] * r1d[1]
		+ 2 * r2d[1] * r2d[1]
		+ r0d[1] * r1d[1]
		+ r0d[1] * r1d[1]
		+ r0d[1] * r2d[1]
		+ r0d[1] * r2d[1]
		+ r1d[1] * r2d[1]
		+ r1d[1] * r2d[1]);

	P_zz += dv / 20 * (2 * r0d[2] * r0d[2]
		+ 2 * r1d[2] * r1d[2]
		+ 2 * r2d[2] * r2d[2]
		+ r0d[2] * r1d[2]
		+ r0d[2] * r1d[2]
		+ r0d[2] * r2d[2]
		+ r0d[2] * r2d[2]
		+ r1d[2] * r2d[2]
		+ r1d[2] * r2d[2]);

	P_xy += dv / 20 * (2 * r0d[0] * r0d[1]
		+ 2 * r1d[0] * r1d[1]
		+ 2 * r2d[0] * r2d[1]
		+ r0d[0] * r1d[1]
		+ r0d[1] * r1d[0]
		+ r0d[0] * r2d[1]
		+ r0d[1] * r2d[0]
		+ r1d[0] * r2d[1]
		+ r1d[1] * r2d[0]);

	P_xz += dv / 20 * (2 * r0d[0] * r0d[2]
		+ 2 * r1d[0] * r1d[2]
		+ 2 * r2d[0] * r2d[2]
		+ r0d[0] * r1d[2]
		+ r0d[2] * r1d[0]
		+ r0d[0] * r2d[2]
		+ r0d[2] * r2d[0]
		+ r1d[0] * r2d[2]
		+ r1d[2] * r2d[0]);

	P_yz += dv / 20 * (2 * r0d[1] * r0d[2]
		+ 2 * r1d[1] * r1d[2]
		+ 2 * r2d[1] * r2d[2]
		+ r0d[1] * r1d[2]
		+ r0d[2] * r1d[1]
		+ r0d[1] * r2d[2]
		+ r0d[2] * r2d[1]
		+ r1d[1] * r2d[2]
		+ r1d[2] * r2d[1]);

}

	// The inertia tensor is finally assembled

arma::mat I = {
	{P_yy + P_zz, -P_xy, -P_xz},
	{ -P_xy, P_xx + P_zz, -P_yz},
	{ -P_xz, -P_yz, P_xx + P_yy}
};

this -> inertia = I;
}




void ShapeModelTri::compute_surface_area() {
	double surface_area = 0;

	#pragma omp parallel for reduction(+:surface_area) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0; facet_index < this -> elements.size(); ++facet_index) {

		Facet * facet = dynamic_cast<Facet *>(this -> elements[facet_index].get());

		surface_area += facet -> get_area();

	}

	this -> surface_area = surface_area;

}




















// void ShapeModelTri::split_facet(Facet * facet,
//                              std::set<Facet *> & seen_facets) {

// 	// The old elements are retrieved
// 	// together with the old vertices

// 	std::set<Facet *> splitted_facets = facet -> get_neighbors(false);

// 	Facet * F1_old = nullptr;
// 	Facet * F2_old = nullptr;
// 	Facet * F3_old = nullptr;


// 	std::vector<std::shared_ptr< ControlPoint> > * V_in_F0_old = facet -> get_control_points();

// 	std::shared_ptr<ControlPoint> V0 = V_in_F0_old -> at(0);
// 	std::shared_ptr<ControlPoint> V1 = V_in_F0_old -> at(1);
// 	std::shared_ptr<ControlPoint> V2 = V_in_F0_old -> at(2);


// 	std::shared_ptr<ControlPoint> V3 ;
// 	std::shared_ptr<ControlPoint> V4 ;
// 	std::shared_ptr<ControlPoint> V5 ;

// 	for (auto const & old_facet : splitted_facets) {
// 		if (old_facet != facet) {
// 			if (V0 -> is_owned_by(old_facet) &&
// 			        V1 -> is_owned_by(old_facet)) {
// 				F1_old = old_facet;
// 				// Setting V4
// 				for (unsigned int i = 0; i < 3; ++i) {
// 					if (F1_old -> get_control_points() -> at(i) != V0 &&
// 					        F1_old -> get_control_points() -> at(i) != V1) {
// 						V4 = F1_old -> get_control_points() -> at(i);
// 						break;
// 					}
// 				}
// 			}
// 			else if (V1 -> is_owned_by(old_facet) &&
// 			         V2 -> is_owned_by(old_facet)) {
// 				F2_old = old_facet;
// 				// Setting V5
// 				for (unsigned int i = 0; i < 3; ++i) {
// 					if (F2_old -> get_control_points() -> at(i) != V2 &&
// 					        F2_old -> get_control_points() -> at(i) != V1) {
// 						V5 = F2_old -> get_control_points() -> at(i);
// 						break;
// 					}
// 				}
// 			}
// 			else if (V2 -> is_owned_by(old_facet) &&
// 			         V0 -> is_owned_by(old_facet)) {
// 				F3_old = old_facet;

// 				// Setting V3
// 				for (unsigned int i = 0; i < 3; ++i) {
// 					if (F3_old -> get_control_points() -> at(i) != V2 &&
// 					        F3_old -> get_control_points() -> at(i) != V0) {
// 						V3 = F3_old -> get_control_points() -> at(i);
// 						break;
// 					}
// 				}
// 			}
// 		}
// 	}


// 	if (F1_old == nullptr || F2_old == nullptr || F3_old == nullptr) {
// 		throw (std::runtime_error("Dangling facet pointers"));
// 	}

// 	// The new vertices and their (empty) coordinates are created
// 	std::shared_ptr<arma::vec> P6 = std::make_shared<arma::vec>(arma::vec(3));
// 	std::shared_ptr<ControlPoint> V6 = std::make_shared<ControlPoint>(ControlPoint());
// 	V6 -> set_coordinates(P6);

// 	std::shared_ptr<arma::vec>  P7 = std::make_shared<arma::vec>(arma::vec(3));
// 	std::shared_ptr<ControlPoint> V7 = std::make_shared<ControlPoint>(ControlPoint());
// 	V7 -> set_coordinates(P7);

// 	std::shared_ptr<arma::vec>  P8 = std::make_shared<arma::vec>(arma::vec(3));
// 	std::shared_ptr<ControlPoint> V8 = std::make_shared<ControlPoint>(ControlPoint());
// 	V8 -> set_coordinates(P8);


// 	// Their coordinates are set to the midpoints of the existing edges
// 	*P6 = ( *V1 -> get_coordinates() + *V2 -> get_coordinates() ) / 2;
// 	*P7 = ( *V0 -> get_coordinates() + *V2 -> get_coordinates() ) / 2;
// 	*P8 = ( *V0 -> get_coordinates() + *V1 -> get_coordinates() ) / 2;


// 	// The new vertices are added to the shape model
// 	this -> add_control_point(V6);
// 	this -> add_control_point(V7);
// 	this -> add_control_point(V8);

// 	// The vertices are grouped accordingly
// 	// to the indexing specified in ShapeModelTri.hpp
// 	// This ensures the consistency of the surface normals

// 	std::vector<std::shared_ptr<ControlPoint>> F0_vertices;
// 	F0_vertices.push_back(V0);
// 	F0_vertices.push_back(V8);
// 	F0_vertices.push_back(V7);

// 	std::vector<std::shared_ptr<ControlPoint>> F1_vertices;
// 	F1_vertices.push_back(V7);
// 	F1_vertices.push_back(V3);
// 	F1_vertices.push_back(V0);

// 	std::vector<std::shared_ptr<ControlPoint>> F2_vertices;
// 	F2_vertices.push_back(V2);
// 	F2_vertices.push_back(V3);
// 	F2_vertices.push_back(V7);

// 	std::vector<std::shared_ptr<ControlPoint>> F3_vertices;
// 	F3_vertices.push_back(V6);
// 	F3_vertices.push_back(V2);
// 	F3_vertices.push_back(V7);

// 	std::vector<std::shared_ptr<ControlPoint>> F4_vertices;
// 	F4_vertices.push_back(V1);
// 	F4_vertices.push_back(V6);
// 	F4_vertices.push_back(V8);

// 	std::vector<std::shared_ptr<ControlPoint>> F5_vertices;
// 	F5_vertices.push_back(V6);
// 	F5_vertices.push_back(V7);
// 	F5_vertices.push_back(V8);

// 	std::vector<std::shared_ptr<ControlPoint>> F6_vertices;
// 	F6_vertices.push_back(V1);
// 	F6_vertices.push_back(V8);
// 	F6_vertices.push_back(V4);

// 	std::vector<std::shared_ptr<ControlPoint>> F7_vertices;
// 	F7_vertices.push_back(V0);
// 	F7_vertices.push_back(V4);
// 	F7_vertices.push_back(V8);

// 	std::vector<std::shared_ptr<ControlPoint>> F8_vertices;
// 	F8_vertices.push_back(V1);
// 	F8_vertices.push_back(V5);
// 	F8_vertices.push_back(V6);

// 	std::vector<std::shared_ptr<ControlPoint>> F9_vertices;
// 	F9_vertices.push_back(V5);
// 	F9_vertices.push_back(V2);
// 	F9_vertices.push_back(V6);

// 	// // The new elements are created
// 	Facet * F0 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F0_vertices));
// 	Facet * F1 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F1_vertices));
// 	Facet * F2 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F2_vertices));
// 	Facet * F3 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F3_vertices));
// 	Facet * F4 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F4_vertices));
// 	Facet * F5 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F5_vertices));
// 	Facet * F6 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F6_vertices));
// 	Facet * F7 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F7_vertices));
// 	Facet * F8 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F8_vertices));
// 	Facet * F9 = new Facet(std::make_shared<std::vector<std::shared_ptr<ControlPoint>>>(F9_vertices));

// 	F0 -> set_split_counter(facet -> get_split_count() + 1);
// 	F1 -> set_split_counter(facet -> get_split_count() + 1);
// 	F2 -> set_split_counter(facet -> get_split_count() + 1);
// 	F3 -> set_split_counter(facet -> get_split_count() + 1);
// 	F4 -> set_split_counter(facet -> get_split_count() + 1);
// 	F5 -> set_split_counter(facet -> get_split_count() + 1);
// 	F6 -> set_split_counter(facet -> get_split_count() + 1);
// 	F7 -> set_split_counter(facet -> get_split_count() + 1);
// 	F8 -> set_split_counter(facet -> get_split_count() + 1);
// 	F9 -> set_split_counter(facet -> get_split_count() + 1);

// 	// // The new elements are added to the shape model
// 	this -> add_element(F0);
// 	this -> add_element(F1);
// 	this -> add_element(F2);
// 	this -> add_element(F3);
// 	this -> add_element(F4);
// 	this -> add_element(F5);
// 	this -> add_element(F6);
// 	this -> add_element(F7);
// 	this -> add_element(F8);
// 	this -> add_element(F9);



// 	// The elements that replaced the recycled elements are added back
// 	seen_facets.insert(F0);
// 	seen_facets.insert(F1);
// 	seen_facets.insert(F2);
// 	seen_facets.insert(F3);
// 	seen_facets.insert(F4);
// 	seen_facets.insert(F5);
// 	seen_facets.insert(F6);
// 	seen_facets.insert(F7);
// 	seen_facets.insert(F8);
// 	seen_facets.insert(F9);


// 	// The elements that were seen and split are removed from the set
// 	seen_facets.erase(facet);
// 	seen_facets.erase(F1_old);
// 	seen_facets.erase(F2_old);
// 	seen_facets.erase(F3_old);



// 	// V0, V1, V2, V3, V4 and V5 still are still
// 	// owned by $facet and its neighbors. Must remove this ownership (note that
// 	// the new ownerships have already been created when constructing
// 	// the new elements)

// 	V0 -> remove_facet_ownership(facet);
// 	V1 -> remove_facet_ownership(facet);
// 	V2 -> remove_facet_ownership(facet);

// 	V0 -> remove_facet_ownership(F1_old);
// 	V0 -> remove_facet_ownership(F3_old);

// 	V1 -> remove_facet_ownership(F1_old);
// 	V1 -> remove_facet_ownership(F2_old);

// 	V2 -> remove_facet_ownership(F2_old);
// 	V2 -> remove_facet_ownership(F3_old);

// 	V4 -> remove_facet_ownership(F1_old);
// 	V5 -> remove_facet_ownership(F2_old);
// 	V3 -> remove_facet_ownership(F3_old);



// 	if (F0 -> get_control_points() -> size() > 3 ||
// 	        F1 -> get_control_points() -> size() > 3 ||
// 	        F2 -> get_control_points() -> size() > 3 ||
// 	        F3 -> get_control_points() -> size() > 3 ||
// 	        F4 -> get_control_points() -> size() > 3 ||
// 	        F5 -> get_control_points() -> size() > 3 ||
// 	        F6 -> get_control_points() -> size() > 3 ||
// 	        F7 -> get_control_points() -> size() > 3 ||
// 	        F8 -> get_control_points() -> size() > 3 ||
// 	        F9 -> get_control_points() -> size() > 3
// 	   ) {
// 		throw (std::runtime_error("One of the new elements has more than three vertices"));
// 	}




// 	// The old elements are deleted and their pointer removed from the shape model
// 	auto old_facet_F0 = std::find (this -> elements.begin(), this -> elements.end(), facet);
// 	delete(*old_facet_F0);
// 	this -> elements.erase(old_facet_F0);

// 	auto old_facet_F1 = std::find (this -> elements.begin(), this -> elements.end(), F1_old);
// 	delete(*old_facet_F1);
// 	this -> elements.erase(old_facet_F1);

// 	auto old_facet_F2 = std::find (this -> elements.begin(), this -> elements.end(), F2_old);
// 	delete(*old_facet_F2);
// 	this -> elements.erase(old_facet_F2);

// 	auto old_facet_F3 = std::find (this -> elements.begin(), this -> elements.end(), F3_old);
// 	delete(*old_facet_F3);
// 	this -> elements.erase(old_facet_F3);


// 	this -> update_mass_properties();

// 	this -> check_normals_consistency(1e-5);

// }



// bool ShapeModelTri::merge_shrunk_facet(double minimum_angle,
//                                     Facet * facet,
//                                     std::set<Facet *> * seen_facets,
//                                     std::set<Facet *> * spurious_facets
//                                    ) {


// 	// The vertices in the facet are extracted
// 	std::shared_ptr<ControlPoint> V0  = facet -> get_control_points() -> at(0);
// 	std::shared_ptr<ControlPoint> V1  = facet -> get_control_points() -> at(1);
// 	std::shared_ptr<ControlPoint> V2  = facet -> get_control_points() -> at(2);


// 	arma::vec * P0  = V0 -> get_coordinates();
// 	arma::vec * P1  = V1 -> get_coordinates();
// 	arma::vec * P2  = V2 -> get_coordinates();

// 	// The smallest of the three angles in the facet is identified
// 	arma::vec angles = arma::vec(3);
// 	angles(0) = std::asin(arma::norm(arma::cross(arma::normalise(*P1 - *P0), arma::normalise(*P2 - *P0))));
// 	angles(1) = std::asin(arma::norm(arma::cross(arma::normalise(*P2 - *P1), arma::normalise(*P0 - *P1))));
// 	angles(2) = std::asin(arma::norm(arma::cross(arma::normalise(*P0 - *P2), arma::normalise(*P1 - *P2))));

// 	if (angles.min() > minimum_angle) {
// 		return false;
// 	}

// 	unsigned int min_angle_index = angles.index_min();

// 	std::shared_ptr<ControlPoint> V_merge_keep = nullptr;
// 	std::shared_ptr<ControlPoint> V_merge_discard = nullptr;

// 	switch (min_angle_index) {

// 	case 0:

// 		V_merge_discard = V1;
// 		V_merge_keep = V2;
// 		break;

// 	case 1:

// 		V_merge_discard = V2;
// 		V_merge_keep = V0;
// 		break;

// 	case 2:
// 		V_merge_discard = V0;
// 		V_merge_keep = V1;
// 		break;
// 	}



// 	std::set<Facet *> facets_to_recycle = V_merge_keep -> common_facets(V_merge_discard);


// 	std::set<std::shared_ptr<ControlPoint> > vertices_to_keep;

// 	for (auto it = facets_to_recycle.begin(); it != facets_to_recycle.end(); ++it) {

// 		for (unsigned int i = 0; i < 3; ++i) {
// 			if ((*it) -> get_control_points() -> at(i) != V_merge_discard && (*it) -> get_control_points() -> at(i) != V_merge_keep) {
// 				vertices_to_keep.insert((*it) -> get_control_points() -> at(i));
// 				break;
// 			}
// 		}
// 	}





// 	std::set<Facet *> facets_to_keep_owning_V_merge_discard = V_merge_discard -> get_owning_elements();
// 	for (auto it = facets_to_recycle.begin(); it != facets_to_recycle.end(); ++it) {
// 		facets_to_keep_owning_V_merge_discard.erase(*it);
// 	}


// 	// If any of the elements to be updated was not seen, the method does not proceed
// 	// if (spurious_facets == nullptr) {
// 	// 	for (auto facet_it = facets_to_keep_owning_V_merge_discard.begin();
// 	// 	        facet_it != facets_to_keep_owning_V_merge_discard.end();
// 	// 	        ++facet_it) {

// 	// 		if (seen_facets -> find(*facet_it) == seen_facets -> end()) {
// 	// 			std::cout << "Connected facet is invisible. Recycling aborted" << std::endl;
// 	// 			return false;
// 	// 		}
// 	// 	}
// 	// }


// 	// If any of the vertices to keep is on a corner (owned by four elements or less), nothing happens
// 	if (
// 	    V_merge_keep -> get_number_of_owning_elements() <= 4 ||
// 	    V_merge_discard -> get_number_of_owning_elements() <= 4) {
// 		return false;
// 	}

// 	for (auto it = vertices_to_keep.begin(); it != vertices_to_keep.end(); ++it) {
// 		if ((*it) -> get_number_of_owning_elements() <= 4) {
// 			return false;
// 		}
// 	}

// 	// The two vertices to be merged are moved to their edge's midpoint
// 	*V_merge_keep -> get_coordinates() = 0.5 * (*V_merge_keep -> get_coordinates()
// 	                                     + *V_merge_discard -> get_coordinates());


// 	// The elements owning V_merge_discard are
// 	// updated so as to have this vertex merging with
// 	// V_merge_keep
// 	for (auto facet_it = facets_to_keep_owning_V_merge_discard.begin();
// 	        facet_it != facets_to_keep_owning_V_merge_discard.end();
// 	        ++facet_it) {

// 		Facet * facet_to_update = *facet_it;


// 		facet_to_update -> recycle_count +=  1;


// 		bool vertex_found = false;
// 		for (unsigned int vertex_index = 0; vertex_index < 3; ++vertex_index) {

// 			if (facet_to_update -> get_control_points() -> at(vertex_index) == V_merge_discard) {

// 				facet_to_update -> get_control_points() -> at(vertex_index) = V_merge_keep;
// 				vertex_found = true;
// 				break;
// 			}

// 		}

// 		if (vertex_found == false) {
// 			throw (std::runtime_error("V_merge_discard was never found in this facet"));
// 		}

// 		facet_to_update -> update();

// 		if (facet_to_update -> get_control_points() -> size() != 3) {
// 			throw (std::runtime_error("this updated facet has " + std::to_string(facet -> get_control_points() -> size()) + " vertices"));
// 		}

// 		/**
// 		I think this is where the bug is
// 		*/
// 		V_merge_discard -> remove_facet_ownership(facet_to_update);
// 		V_merge_keep -> add_element_ownership(facet_to_update);

// 	}



// 	for (auto it_f = facets_to_recycle.begin(); it_f != facets_to_recycle.end(); ++it_f) {

// 		for (auto it_v = vertices_to_keep.begin(); it_v != vertices_to_keep.end(); ++it_v) {
// 			(*it_v) -> remove_facet_ownership(*it_f);
// 		}

// 		V_merge_keep -> remove_facet_ownership(*it_f);
// 		V_merge_discard -> remove_facet_ownership(*it_f);

// 		seen_facets -> erase(*it_f);

// 		if (spurious_facets != nullptr) {
// 			spurious_facets -> erase(*it_f);
// 		}


// 		// The elements to recycle are removed from the shape model
// 		auto old_facet = std::find (this -> elements.begin(), this -> elements.end(), *it_f);
// 		delete(*old_facet);
// 		this -> elements.erase(old_facet);

// 	}


// 	// The discarded vertex is removed from the shape model
// 	auto V_merge_discard_it = std::find (this -> vertices.begin(),
// 	                                     this -> vertices.end(),
// 	                                     V_merge_discard);


// 	if (V_merge_discard_it != this -> vertices.end()) {
// 		this -> vertices.erase(V_merge_discard_it);
// 	}
// 	else {
// 		throw (std::runtime_error("V_merge_discard not found in vertices"));
// 	}


// 	// The impacted elements are all updated to reflect their new geometry
// 	this -> update_facets();

// 	// Check if there are any dangling vertex
// 	if ( V_merge_keep -> get_number_of_owning_elements() < 3 ) {
// 		throw (std::runtime_error("Dangling vertex leaving merge: V_merge_keep was owned by " + std::to_string(V_merge_keep -> get_number_of_owning_elements()) + " elements"));
// 	}


// 	for (auto it_v = vertices_to_keep.begin(); it_v != vertices_to_keep.end(); ++it_v) {
// 		if ( (*it_v) -> get_number_of_owning_elements() < 3 ) {
// 			throw (std::runtime_error("Dangling vertex leaving merge: this vertex was owned by " + std::to_string((*it_v) -> get_number_of_owning_elements()) + " elements"));
// 		}
// 	}

// 	return true;

// }






