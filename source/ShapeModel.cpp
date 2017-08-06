#include "../include/ShapeModel.hpp"
#include <chrono>


ShapeModel::ShapeModel() {

}

void ShapeModel::update_mass_properties() {

	this -> compute_surface_area();
	this -> compute_volume();
	this -> compute_center_of_mass();
	// this -> compute_inertia();

}

void ShapeModel::update_facets(bool compute_dyad) {

	for (auto & facet : this -> facets) {
		facet -> update(compute_dyad);
	}

}


void ShapeModel::update_edges() {

	for (auto & edge : this -> edges) {
		edge -> compute_dyad();
	}

}



void ShapeModel::update_facets(std::set<Facet *> & facets, bool compute_dyad) {

	for (auto & facet : facets) {
		facet -> update(compute_dyad);
	}

}


bool ShapeModel::has_kd_tree() const {
	if (this -> kd_tree == nullptr) {
		return false;
	}
	else {
		return true;
	}
}

std::shared_ptr<KDTree_Shape> ShapeModel::get_kdtree() const {
	return this -> kd_tree;
}

void ShapeModel::construct_kd_tree(bool verbose) {

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();


	this -> kd_tree = std::make_shared<KDTree_Shape>(KDTree_Shape());
	this -> kd_tree = this -> kd_tree -> build(this -> facets, 0, verbose);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;


	std::cout << "\n Elapsed time during KDTree construction : " << elapsed_seconds.count() << "s\n\n";

}


bool ShapeModel::contains(double * point, double tol ) {

	double lagrangian = 0;

	// Facet loop
	#pragma omp parallel for reduction(+:lagrangian) if (USE_OMP_DYNAMIC_ANALYSIS)
	for (unsigned int facet_index = 0; facet_index < this -> get_NFacets(); ++ facet_index) {

		std::vector<std::shared_ptr<Vertex > > * vertices = this -> get_facets() -> at(facet_index) -> get_vertices();

		const double * r1 =  vertices -> at(0) -> get_coordinates() -> colptr(0);
		const double * r2 =  vertices -> at(1) -> get_coordinates() -> colptr(0);
		const double * r3 =  vertices -> at(2) -> get_coordinates() -> colptr(0);

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


void ShapeModel::save(std::string path) const {
	std::ofstream shape_file;
	shape_file.open(path);

	std::map<std::shared_ptr<Vertex> , unsigned int> vertex_ptr_to_index;

	for (unsigned int vertex_index = 0;
	        vertex_index < this -> get_NVertices();
	        ++vertex_index) {

		shape_file << "v " << this -> vertices[vertex_index] -> get_coordinates() -> colptr(0)[0] << " " << this -> vertices[vertex_index] -> get_coordinates() -> colptr(0)[1] << " " << this -> vertices[vertex_index] -> get_coordinates() -> colptr(0)[2] << std::endl;
		vertex_ptr_to_index[this -> vertices[vertex_index]] = vertex_index;
	}

	for (unsigned int facet_index = 0;
	        facet_index < this -> get_NFacets();
	        ++facet_index) {

		unsigned int v0 =  vertex_ptr_to_index[this -> facets[facet_index] -> get_vertices() -> at(0)] + 1;
		unsigned int v1 =  vertex_ptr_to_index[this -> facets[facet_index] -> get_vertices() -> at(1)] + 1;
		unsigned int v2 =  vertex_ptr_to_index[this -> facets[facet_index] -> get_vertices() -> at(2)] + 1;

		shape_file << "f " << v0 << " " << v1 << " " << v2 << std::endl;

	}




	shape_file.close();




}


void ShapeModel::shift_to_barycenter() {

	arma::vec x = - (*this -> get_center_of_mass());

	// The vertices are shifted
	#pragma omp parallel for if(USE_OMP_SHAPE_MODEL)
	for (unsigned int vertex_index = 0;
	        vertex_index < this -> get_NVertices();
	        ++vertex_index) {

		*this -> vertices[vertex_index] -> get_coordinates() = *this -> vertices[vertex_index] -> get_coordinates() + x;

	}

	this -> cm = 0 * this -> cm;

}

void ShapeModel::align_with_principal_axes() {

	arma::vec moments;
	arma::mat axes ;

	this -> compute_inertia();

	double T = arma::trace(this -> inertia) ;
	double Pi = 0.5 * (T * T - arma::trace(this -> inertia * this -> inertia));
	double U = std::sqrt(T * T - 3 * Pi) / 3;
	double Det = arma::det(this -> inertia);

	std::cout << "Non-dimensional inertia: " << std::endl;
	std::cout << this -> inertia << std::endl;


	if (U > 1e-6) {

		double cos_Theta = (- 2 * T * T * T +  9 * T * Pi - 27 * Det) / (54 * U * U * U );
		double Theta;

		if (cos_Theta > 0) {

		}

		if (std::abs(std::abs(cos_Theta) - 1) < 1e-6) {
			if (cos_Theta > 0) {
				Theta = 0;
			}
			else {
				Theta = arma::datum::pi;
			}
		}
		else {
			Theta = std::acos( (- 2 * T * T * T +  9 * T * Pi - 27 * Det) / (54 * U * U * U ));
		}



		double A = T / 3 - 2 * U * std::cos(Theta / 3);
		double B = T / 3 - 2 * U * std::cos(Theta / 3 - 2 * arma::datum::pi / 3);
		double C = T / 3 - 2 * U * std::cos(Theta / 3 + 2 * arma::datum::pi / 3);

		moments = {A, B, C};


		arma::mat L0 = this -> inertia - moments(0) * arma::eye<arma::mat>(3, 3);
		arma::mat L1 = this -> inertia - moments(1) * arma::eye<arma::mat>(3, 3);

		L0.row(0) = arma::normalise(L0.row(0));
		L0.row(1) = arma::normalise(L0.row(1));
		L0.row(2) = arma::normalise(L0.row(2));

		L1.row(0) = arma::normalise(L1.row(0));
		L1.row(1) = arma::normalise(L1.row(1));
		L1.row(2) = arma::normalise(L1.row(2));

		arma::mat e0_mat(3, 3);

		e0_mat.row(0) = arma::cross(L0.row(0), L0.row(1));
		e0_mat.row(1) = arma::cross(L0.row(0), L0.row(2));
		e0_mat.row(2) = arma::cross(L0.row(1), L0.row(2));

		arma::vec norms_e0 = {arma::norm(e0_mat.row(0)), arma::norm(e0_mat.row(1)), arma::norm(e0_mat.row(2))};
		double best_e0 = norms_e0.index_max();
		arma::vec e0 = arma::normalise(e0_mat.row(best_e0).t());

		arma::mat e1_mat(3, 3);
		e1_mat.row(0) = arma::cross(L1.row(0), L1.row(1));
		e1_mat.row(1) = arma::cross(L1.row(0), L1.row(2));
		e1_mat.row(2) = arma::cross(L1.row(1), L1.row(2));

		arma::vec norms_e1 = {arma::norm(e1_mat.row(0)), arma::norm(e1_mat.row(1)), arma::norm(e1_mat.row(2))};
		double best_e1 = norms_e1.index_max();
		arma::vec e1 = arma::normalise(e1_mat.row(best_e1).t());

		arma::vec e2 = arma::cross(e0, e1);

		axes = arma::join_rows(e0, arma::join_rows(e1, e2));
	}

	else {
		moments = std::pow(Det, 1. / 3.) * arma::ones<arma::vec>(3);
		axes = arma::eye<arma::mat>(3, 3);
	}

	std::cout << "Principal axes: " << std::endl;
	std::cout << axes << std::endl;

	std::cout << "Non-dimensional principal moments: " << std::endl;
	std::cout << moments << std::endl;

	// The vertices are shifted
	#pragma omp parallel for if(USE_OMP_SHAPE_MODEL)
	for (unsigned int vertex_index = 0;
	        vertex_index < this -> get_NVertices();
	        ++vertex_index) {

		*this -> vertices[vertex_index] -> get_coordinates() = axes.t() * (*this -> vertices[vertex_index] -> get_coordinates());
	}

	this -> inertia = arma::diagmat(moments);

}

arma::mat ShapeModel::get_inertia() const {
	return this -> inertia;

}



ShapeModel::ShapeModel(std::string ref_frame_name,
                       FrameGraph * frame_graph) {
	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
}



void ShapeModel::add_facet(Facet * facet) {
	this -> facets. push_back(facet);
}

void ShapeModel::save_lat_long_map_to_file(std::string path) const {

	arma::mat long_lat_hit_count = arma::mat(this -> get_NFacets(), 3);

	for (unsigned int facet_index = 0; facet_index < this -> get_NFacets(); ++facet_index) {

		Facet * facet = this -> facets . at(facet_index);
		arma::vec * facet_center = facet -> get_facet_center();

		double longitude = 180. / arma::datum::pi * std::atan2(facet_center -> at(1), facet_center -> at(0)) ;
		double latitude = 180. / arma::datum::pi * std::atan2(facet_center -> at(2), arma::norm(facet_center ->rows(0, 1)));
		unsigned int hit_count = facet -> get_hit_count();
		arma::rowvec facet_results = {longitude, latitude, (double)(hit_count)};

		long_lat_hit_count.row(facet_index) = facet_results;

	}

	long_lat_hit_count.save(path, arma::raw_ascii);

}


std::string ShapeModel::get_ref_frame_name() const {
	return this -> ref_frame_name;
}


void ShapeModel::add_edge(std::shared_ptr<Edge> edge) {
	this -> edges. push_back(edge);
}

void ShapeModel::add_vertex(std::shared_ptr<Vertex> vertex) {
	this -> vertices.push_back(vertex);
}

ShapeModel::~ShapeModel() {
	for (unsigned int facet_index = 0; facet_index < this -> facets.size(); ++ facet_index) {
		delete(this -> facets[facet_index]);
	}
}

unsigned int ShapeModel::get_NFacets() const {
	return this -> facets . size();
}

unsigned int ShapeModel::get_NVertices() const {
	return this -> vertices . size();
}

unsigned int ShapeModel::get_NEdges() const {
	return this -> edges . size();
}

std::vector<std::shared_ptr< Vertex> > * ShapeModel::get_vertices() {
	return &this -> vertices;
}


std::vector<Facet * > * ShapeModel::get_facets() {
	return &this -> facets;
}

std::vector<std::shared_ptr< Edge> > * ShapeModel::get_edges() {
	return &this -> edges;
}


void ShapeModel::check_normals_consistency(double tol) const {
	double facet_area_average = 0;

	double sx = 0;
	double sy = 0;
	double sz = 0;

	#pragma omp parallel for reduction(+:facet_area_average,sx,sy,sz) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0; facet_index < this -> facets.size(); ++facet_index) {

		Facet * facet = this -> facets[facet_index];

		sx += facet -> get_area() * facet -> get_facet_normal() -> at(0);
		sy += facet -> get_area() * facet -> get_facet_normal() -> at(1);
		sz += facet -> get_area() * facet -> get_facet_normal() -> at(2);

		facet_area_average += facet -> get_area();

	}

	arma::vec surface_sum = {sx, sy, sz};

	facet_area_average = facet_area_average / this -> facets.size();
	if (arma::norm(surface_sum) / facet_area_average > tol) {
		throw (std::runtime_error("Normals were incorrectly oriented. norm(sum(n * s))/sum(s)= " + std::to_string(arma::norm(surface_sum) / facet_area_average)));
	}

}



void ShapeModel::compute_volume() {
	double volume = 0;

	#pragma omp parallel for reduction(+:volume) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;
	        facet_index < this -> facets.size();
	        ++facet_index) {

		std::vector<std::shared_ptr<Vertex > > * vertices = this -> facets[facet_index] -> get_vertices();

		arma::vec * r0 =  vertices -> at(0) -> get_coordinates();
		arma::vec * r1 =  vertices -> at(1) -> get_coordinates();
		arma::vec * r2 =  vertices -> at(2) -> get_coordinates();
		double dv = arma::dot(*r0, arma::cross(*r1 - *r0, *r2 - *r0)) / 6.;
		volume = volume + dv;

	}

	this -> volume = volume;

}




void ShapeModel::compute_center_of_mass() {
	double c_x = 0;
	double c_y = 0;
	double c_z = 0;
	double volume = this -> get_volume();

	#pragma omp parallel for reduction(+:c_x,c_y,c_z) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;
	        facet_index < this -> facets.size();
	        ++facet_index) {


		std::vector<std::shared_ptr<Vertex > > * vertices = this -> facets[facet_index] -> get_vertices();

		arma::vec * r0 =  vertices -> at(0) -> get_coordinates();
		arma::vec * r1 =  vertices -> at(1) -> get_coordinates();
		arma::vec * r2 =  vertices -> at(2) -> get_coordinates();

		double * r0d =  vertices -> at(0) -> get_coordinates() -> colptr(0);
		double * r1d =  vertices -> at(1) -> get_coordinates() -> colptr(0);
		double * r2d =  vertices -> at(2) -> get_coordinates() -> colptr(0);

		double dv = 1. / 6. * arma::dot(*r1, arma::cross(*r1 - *r0, *r2 - *r0));

		double dr_x = (r0d[0] + r1d[0] + r2d[0]) / 4.;
		double dr_y = (r0d[1] + r1d[1] + r2d[1]) / 4.;
		double dr_z = (r0d[2] + r1d[2] + r2d[2]) / 4.;

		c_x = c_x + dv * dr_x / volume;
		c_y = c_y + dv * dr_y / volume;
		c_z = c_z + dv * dr_z / volume;

	}

	arma::vec cm = {c_x, c_y, c_z};

	this -> cm =  cm ;


}


void ShapeModel::compute_inertia() {


	double P_xx = 0;
	double P_yy = 0;
	double P_zz = 0;
	double P_xy = 0;
	double P_xz = 0;
	double P_yz = 0;

	double l = std::pow(this -> volume, 1. / 3.);

	#pragma omp parallel for reduction(+:P_xx,P_yy,P_zz,P_xy,P_xz,P_yz) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;
	        facet_index < this -> facets.size();
	        ++facet_index) {


		std::vector<std::shared_ptr<Vertex > > * vertices = this -> facets[facet_index] -> get_vertices();

		// Normalized coordinates
		arma::vec r0 =  (*vertices -> at(0) -> get_coordinates()) / l;
		arma::vec r1 =  (*vertices -> at(1) -> get_coordinates()) / l;
		arma::vec r2 =  (*vertices -> at(2) -> get_coordinates()) / l;

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


double ShapeModel::get_volume() const {
	return this -> volume;
}


double ShapeModel::get_surface_area() const {
	return this -> surface_area;
}


arma::vec * ShapeModel::get_center_of_mass() {
	return &(this -> cm);
}


void ShapeModel::compute_surface_area() {
	double surface_area = 0;

	#pragma omp parallel for reduction(+:surface_area) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0; facet_index < this -> facets.size(); ++facet_index) {

		Facet * facet = this -> facets[facet_index];

		surface_area += facet -> get_area();


	}

	this -> surface_area = surface_area;

}




















void ShapeModel::split_facet(Facet * facet,
                             std::set<Facet *> & seen_facets) {

	// The old facets are retrieved
	// together with the old vertices

	std::set<Facet *> splitted_facets = facet -> get_neighbors(false);

	Facet * F1_old = nullptr;
	Facet * F2_old = nullptr;
	Facet * F3_old = nullptr;


	std::vector<std::shared_ptr< Vertex> > * V_in_F0_old = facet -> get_vertices();

	std::shared_ptr<Vertex> V0 = V_in_F0_old -> at(0);
	std::shared_ptr<Vertex> V1 = V_in_F0_old -> at(1);
	std::shared_ptr<Vertex> V2 = V_in_F0_old -> at(2);


	std::shared_ptr<Vertex> V3 ;
	std::shared_ptr<Vertex> V4 ;
	std::shared_ptr<Vertex> V5 ;

	for (auto const & old_facet : splitted_facets) {
		if (old_facet != facet) {
			if (V0 -> is_owned_by(old_facet) &&
			        V1 -> is_owned_by(old_facet)) {
				F1_old = old_facet;
				// Setting V4
				for (unsigned int i = 0; i < 3; ++i) {
					if (F1_old -> get_vertices() -> at(i) != V0 &&
					        F1_old -> get_vertices() -> at(i) != V1) {
						V4 = F1_old -> get_vertices() -> at(i);
						break;
					}
				}
			}
			else if (V1 -> is_owned_by(old_facet) &&
			         V2 -> is_owned_by(old_facet)) {
				F2_old = old_facet;
				// Setting V5
				for (unsigned int i = 0; i < 3; ++i) {
					if (F2_old -> get_vertices() -> at(i) != V2 &&
					        F2_old -> get_vertices() -> at(i) != V1) {
						V5 = F2_old -> get_vertices() -> at(i);
						break;
					}
				}
			}
			else if (V2 -> is_owned_by(old_facet) &&
			         V0 -> is_owned_by(old_facet)) {
				F3_old = old_facet;

				// Setting V3
				for (unsigned int i = 0; i < 3; ++i) {
					if (F3_old -> get_vertices() -> at(i) != V2 &&
					        F3_old -> get_vertices() -> at(i) != V0) {
						V3 = F3_old -> get_vertices() -> at(i);
						break;
					}
				}
			}
		}
	}


	if (F1_old == nullptr || F2_old == nullptr || F3_old == nullptr) {
		throw (std::runtime_error("Dangling facet pointers"));
	}

	// The new vertices and their (empty) coordinates are created
	std::shared_ptr<arma::vec> P6 = std::make_shared<arma::vec>(arma::vec(3));
	std::shared_ptr<Vertex> V6 = std::make_shared<Vertex>(Vertex());
	V6 -> set_coordinates(P6);

	std::shared_ptr<arma::vec>  P7 = std::make_shared<arma::vec>(arma::vec(3));
	std::shared_ptr<Vertex> V7 = std::make_shared<Vertex>(Vertex());
	V7 -> set_coordinates(P7);

	std::shared_ptr<arma::vec>  P8 = std::make_shared<arma::vec>(arma::vec(3));
	std::shared_ptr<Vertex> V8 = std::make_shared<Vertex>(Vertex());
	V8 -> set_coordinates(P8);


	// Their coordinates are set to the midpoints of the existing edges
	*P6 = ( *V1 -> get_coordinates() + *V2 -> get_coordinates() ) / 2;
	*P7 = ( *V0 -> get_coordinates() + *V2 -> get_coordinates() ) / 2;
	*P8 = ( *V0 -> get_coordinates() + *V1 -> get_coordinates() ) / 2;


	// The new vertices are added to the shape model
	this -> add_vertex(V6);
	this -> add_vertex(V7);
	this -> add_vertex(V8);

	// The vertices are grouped accordingly
	// to the indexing specified in ShapeModel.hpp
	// This ensures the consistency of the surface normals

	std::vector<std::shared_ptr<Vertex>> F0_vertices;
	F0_vertices.push_back(V0);
	F0_vertices.push_back(V8);
	F0_vertices.push_back(V7);

	std::vector<std::shared_ptr<Vertex>> F1_vertices;
	F1_vertices.push_back(V7);
	F1_vertices.push_back(V3);
	F1_vertices.push_back(V0);

	std::vector<std::shared_ptr<Vertex>> F2_vertices;
	F2_vertices.push_back(V2);
	F2_vertices.push_back(V3);
	F2_vertices.push_back(V7);

	std::vector<std::shared_ptr<Vertex>> F3_vertices;
	F3_vertices.push_back(V6);
	F3_vertices.push_back(V2);
	F3_vertices.push_back(V7);

	std::vector<std::shared_ptr<Vertex>> F4_vertices;
	F4_vertices.push_back(V1);
	F4_vertices.push_back(V6);
	F4_vertices.push_back(V8);

	std::vector<std::shared_ptr<Vertex>> F5_vertices;
	F5_vertices.push_back(V6);
	F5_vertices.push_back(V7);
	F5_vertices.push_back(V8);

	std::vector<std::shared_ptr<Vertex>> F6_vertices;
	F6_vertices.push_back(V1);
	F6_vertices.push_back(V8);
	F6_vertices.push_back(V4);

	std::vector<std::shared_ptr<Vertex>> F7_vertices;
	F7_vertices.push_back(V0);
	F7_vertices.push_back(V4);
	F7_vertices.push_back(V8);

	std::vector<std::shared_ptr<Vertex>> F8_vertices;
	F8_vertices.push_back(V1);
	F8_vertices.push_back(V5);
	F8_vertices.push_back(V6);

	std::vector<std::shared_ptr<Vertex>> F9_vertices;
	F9_vertices.push_back(V5);
	F9_vertices.push_back(V2);
	F9_vertices.push_back(V6);

	// // The new facets are created
	Facet * F0 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F0_vertices));
	Facet * F1 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F1_vertices));
	Facet * F2 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F2_vertices));
	Facet * F3 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F3_vertices));
	Facet * F4 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F4_vertices));
	Facet * F5 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F5_vertices));
	Facet * F6 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F6_vertices));
	Facet * F7 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F7_vertices));
	Facet * F8 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F8_vertices));
	Facet * F9 = new Facet(std::make_shared<std::vector<std::shared_ptr<Vertex>>>(F9_vertices));

	F0 -> set_split_counter(facet -> get_split_count() + 1);
	F1 -> set_split_counter(facet -> get_split_count() + 1);
	F2 -> set_split_counter(facet -> get_split_count() + 1);
	F3 -> set_split_counter(facet -> get_split_count() + 1);
	F4 -> set_split_counter(facet -> get_split_count() + 1);
	F5 -> set_split_counter(facet -> get_split_count() + 1);
	F6 -> set_split_counter(facet -> get_split_count() + 1);
	F7 -> set_split_counter(facet -> get_split_count() + 1);
	F8 -> set_split_counter(facet -> get_split_count() + 1);
	F9 -> set_split_counter(facet -> get_split_count() + 1);

	// // The new facets are added to the shape model
	this -> add_facet(F0);
	this -> add_facet(F1);
	this -> add_facet(F2);
	this -> add_facet(F3);
	this -> add_facet(F4);
	this -> add_facet(F5);
	this -> add_facet(F6);
	this -> add_facet(F7);
	this -> add_facet(F8);
	this -> add_facet(F9);



	// The facets that replaced the recycled facets are added back
	seen_facets.insert(F0);
	seen_facets.insert(F1);
	seen_facets.insert(F2);
	seen_facets.insert(F3);
	seen_facets.insert(F4);
	seen_facets.insert(F5);
	seen_facets.insert(F6);
	seen_facets.insert(F7);
	seen_facets.insert(F8);
	seen_facets.insert(F9);


	// The facets that were seen and split are removed from the set
	seen_facets.erase(facet);
	seen_facets.erase(F1_old);
	seen_facets.erase(F2_old);
	seen_facets.erase(F3_old);


	/*
	EDGES ARE IGNORED FOR NOW
	*/

	// V0, V1, V2, V3, V4 and V5 still are still
	// owned by $facet and its neighbors. Must remove this ownership (note that
	// the new ownerships have already been created when constructing
	// the new facets)

	V0 -> remove_facet_ownership(facet);
	V1 -> remove_facet_ownership(facet);
	V2 -> remove_facet_ownership(facet);

	V0 -> remove_facet_ownership(F1_old);
	V0 -> remove_facet_ownership(F3_old);

	V1 -> remove_facet_ownership(F1_old);
	V1 -> remove_facet_ownership(F2_old);

	V2 -> remove_facet_ownership(F2_old);
	V2 -> remove_facet_ownership(F3_old);

	V4 -> remove_facet_ownership(F1_old);
	V5 -> remove_facet_ownership(F2_old);
	V3 -> remove_facet_ownership(F3_old);



	if (F0 -> get_vertices() -> size() > 3 ||
	        F1 -> get_vertices() -> size() > 3 ||
	        F2 -> get_vertices() -> size() > 3 ||
	        F3 -> get_vertices() -> size() > 3 ||
	        F4 -> get_vertices() -> size() > 3 ||
	        F5 -> get_vertices() -> size() > 3 ||
	        F6 -> get_vertices() -> size() > 3 ||
	        F7 -> get_vertices() -> size() > 3 ||
	        F8 -> get_vertices() -> size() > 3 ||
	        F9 -> get_vertices() -> size() > 3
	   ) {
		throw (std::runtime_error("One of the new facets has more than three vertices"));
	}




	// The old facets are deleted and their pointer removed from the shape model
	auto old_facet_F0 = std::find (this -> facets.begin(), this -> facets.end(), facet);
	delete(*old_facet_F0);
	this -> facets.erase(old_facet_F0);

	auto old_facet_F1 = std::find (this -> facets.begin(), this -> facets.end(), F1_old);
	delete(*old_facet_F1);
	this -> facets.erase(old_facet_F1);

	auto old_facet_F2 = std::find (this -> facets.begin(), this -> facets.end(), F2_old);
	delete(*old_facet_F2);
	this -> facets.erase(old_facet_F2);

	auto old_facet_F3 = std::find (this -> facets.begin(), this -> facets.end(), F3_old);
	delete(*old_facet_F3);
	this -> facets.erase(old_facet_F3);


	this -> update_mass_properties();

	this -> check_normals_consistency(1e-5);

}



bool ShapeModel::merge_shrunk_facet(Facet * facet,
                                    std::set<Facet *> * seen_facets,
                                    std::set<Facet *> * spurious_facets) {



	std::cout << "In merge_shrunk_facet" << std::endl;


	std::cout << facet -> get_vertices() << std::endl;

	std::cout << "moving on?" << std::endl;

	std::cout << facet -> get_vertices() -> size() << std::endl;

	std::cout << "moving on" << std::endl;

	if (facet -> get_vertices() -> size() != 3) {
		throw (std::runtime_error("this facet has " + std::to_string(facet -> get_vertices() -> size()) + " vertices"));
	}

	// The vertices in the facet are extracted
	std::shared_ptr<Vertex> V0  = facet -> get_vertices() -> at(0);
	std::shared_ptr<Vertex> V1  = facet -> get_vertices() -> at(1);
	std::shared_ptr<Vertex> V2  = facet -> get_vertices() -> at(2);


	std::cout << "Done reading vertices" << std::endl;

	arma::vec * P0  = V0 -> get_coordinates();
	arma::vec * P1  = V1 -> get_coordinates();
	arma::vec * P2  = V2 -> get_coordinates();

	std::cout << "Done reading coordinates" << std::endl;


	// The smallest of the three angles in the facet is identified
	arma::vec angles = arma::vec(3);
	angles(0) = std::acos(abs(arma::dot(arma::normalise(*P1 - *P0), arma::normalise(*P2 - *P0))));
	angles(1) = std::acos(abs(arma::dot(arma::normalise(*P2 - *P1), arma::normalise(*P0 - *P1))));
	angles(2) = std::acos(abs(arma::dot(arma::normalise(*P0 - *P2), arma::normalise(*P1 - *P2))));

	// This index indicates which vertices are to be merged
	// - 0 : V1 and V2 shoud be merged
	// - 1 : V0 and V2
	// - 2 : V1 and V0
	unsigned int min_angle_index = angles.index_min();


	std::shared_ptr<Vertex> V_merge_keep = nullptr;
	std::shared_ptr<Vertex> V_merge_discard = nullptr;
	std::shared_ptr<Vertex> V_keep_0 = nullptr;
	std::shared_ptr<Vertex> V_keep_1 = nullptr;


	switch (min_angle_index) {
	case 0:
		V_merge_keep = V1;
		V_merge_discard = V2;
		break;

	case 1:
		V_merge_keep = V0;
		V_merge_discard = V2;
		break;

	case 2:
		V_merge_keep = V1;
		V_merge_discard = V0;
		break;
	}



	std::vector<Facet *> facets_to_recycle = V_merge_keep -> common_facets(V_merge_discard);

	if (facets_to_recycle.size() > 2) {

		for (unsigned int i = 0; i < facets_to_recycle.size() ; ++i ) {
			std::cout << "Facet at " <<  facets_to_recycle[i] -> get_facet_center() -> t() << std::endl;
			std::cout << "Normal : " <<  facets_to_recycle[i] -> get_facet_normal() -> t() << std::endl;
			std::cout << "Area : " <<  facets_to_recycle[i] -> get_area() << std::endl;



			std::cout << "\t v0 : " <<  facets_to_recycle[i] -> get_vertices() -> at(0) -> get_coordinates() -> t() << std::endl;
			std::cout << "\t v1 : " <<  facets_to_recycle[i] -> get_vertices() -> at(1) -> get_coordinates() -> t() << std::endl;
			std::cout << "\t v2 : " <<  facets_to_recycle[i] -> get_vertices() -> at(2) -> get_coordinates() -> t() << std::endl;


			if (this -> facets.end() != std::find (this -> facets.begin(), this -> facets.end(), facets_to_recycle[i])) {
				std::cout << "Facet still in shape model" << std::endl;
			}
			else {
				std::cout << "Facet no longer in shape model" << std::endl;
			}
		}
		throw (std::runtime_error("Two vertices can't share more than two facets: these shared " + std::to_string(facets_to_recycle.size()) + " facets"));
	}

	Facet * F0_old = facets_to_recycle[0];
	Facet * F1_old = facets_to_recycle[1];

	for (unsigned int vertex_index = 0; vertex_index < 3; ++vertex_index) {
		if (F0_old -> get_vertices() -> at(vertex_index) != V_merge_keep &&
		        F0_old -> get_vertices() -> at(vertex_index) != V_merge_discard) {

			V_keep_0 = F0_old -> get_vertices() -> at(vertex_index);
			break;
		}
	}

	for (unsigned int vertex_index = 0; vertex_index < 3; ++vertex_index) {


		if (F1_old -> get_vertices() -> at(vertex_index) != V_merge_keep &&
		        F1_old -> get_vertices() -> at(vertex_index) != V_merge_discard) {
			V_keep_1 = F1_old -> get_vertices() -> at(vertex_index);

			break;
		}
	}

	if (V_keep_0 == nullptr || V_keep_1 == nullptr) {
		std::cout << V_keep_0 << std::endl;
		std::cout << V_keep_1 << std::endl;

		throw (std::runtime_error("Null pointer floating around"));
	}


	std::set<Facet * > facets_owning_discarded_vertex = V_merge_discard -> get_owning_facets();

	if (facets_owning_discarded_vertex.find(F0_old) == facets_owning_discarded_vertex.end()) {
		throw (std::runtime_error("F0_old not in facets_owning_discarded_vertex"));
	}


	if (facets_owning_discarded_vertex.find(F1_old) == facets_owning_discarded_vertex.end()) {
		throw (std::runtime_error("F1_old not in facets_owning_discarded_vertex"));
	}

	// If any of the facets to be updated was not seen, the method does not proceed
	if (spurious_facets == nullptr) {
		for (auto facet_it = facets_owning_discarded_vertex.begin();
		        facet_it != facets_owning_discarded_vertex.end();
		        ++facet_it) {

			if (seen_facets -> find(*facet_it) == seen_facets -> end()) {
				std::cout << "Connected facet is invisible. Recycling aborted" << std::endl;
				return false;
			}
		}
	}

	// If any of the vertices to keep is on a corner (owned by three facets), nothing happens
	if (V_keep_0 -> get_number_of_owning_facets() == 3 || V_keep_1 -> get_number_of_owning_facets() == 3 ) {
		return false;
	}


	if (facet != F0_old && facet != F1_old) {
		throw (std::runtime_error("facet should be either of these"));
	}

	if (F0_old == F1_old) {
		throw (std::runtime_error("These two facets can't be the same!"));
	}


	// The facets that will be discarded are removed from these facet set
	unsigned int size_facets_owning_discarded_vertex_before = facets_owning_discarded_vertex.size();
	facets_owning_discarded_vertex.erase(F0_old);
	facets_owning_discarded_vertex.erase(F1_old);

	if (size_facets_owning_discarded_vertex_before - 2 != facets_owning_discarded_vertex.size()) {
		throw (std::runtime_error("owning facet removal failed: difference = " + std::to_string(size_facets_owning_discarded_vertex_before - facets_owning_discarded_vertex.size())));
	}


	std::cout << "Recycling" << std::endl;

	*V_merge_keep -> get_coordinates() = 0.5 * (*V_merge_keep -> get_coordinates() + *V_merge_discard -> get_coordinates());


	// The facets owning V_merge_discard are
	// updated so as to have this vertex merging with
	// V_merge_keep


	// Check if there are any dangling vertex
	for (auto facet_it = this -> get_facets() -> begin();
	        facet_it != this -> get_facets() -> end();
	        ++facet_it) {
		for (unsigned int vertex_index = 0;
		        vertex_index < 3; ++vertex_index) {

			if ( (*facet_it) -> get_vertices() -> at(vertex_index) -> get_number_of_owning_facets() < 3 ) {
				throw (std::runtime_error("Dangling vertex entering merge"));
			}

		}
	}


	std::cout << "browsing facets" << std::endl;
	for (auto facet_it = facets_owning_discarded_vertex.begin();
	        facet_it != facets_owning_discarded_vertex.end();
	        ++facet_it) {

		Facet * facet_to_update = *facet_it;




		for (unsigned int vertex_index = 0; vertex_index < 3; ++vertex_index) {

			if (facet_to_update -> get_vertices() -> at(vertex_index) == V_merge_discard) {

				facet_to_update -> get_vertices() -> at(vertex_index) = V_merge_keep;
				break;
			}

		}


		facet_to_update -> update(false);



		if (facet_to_update -> get_vertices() -> size() != 3) {
			throw (std::runtime_error("this updated facet has " + std::to_string(facet -> get_vertices() -> size()) + " vertices"));
		}



		V_merge_discard -> remove_facet_ownership(facet_to_update);
		V_merge_keep -> add_facet_ownership(facet_to_update);

	}

	std::cout << "Done browsing facets" << std::endl;

	if (V_merge_keep == V_merge_discard) {
		throw std::runtime_error("V_merge_discard and V_merge_keep are the same");
	}






	// V_keep_0,1 and V_merge_keep are still owned by the facets to be recycled
	V_keep_0 -> remove_facet_ownership(F0_old);
	V_keep_1 -> remove_facet_ownership(F1_old);
	V_merge_keep -> remove_facet_ownership(F0_old);
	V_merge_keep -> remove_facet_ownership(F1_old);
	V_merge_discard -> remove_facet_ownership(F0_old);
	V_merge_discard -> remove_facet_ownership(F1_old);



	std::cout << "Looking for discarded vertex" << std::endl;

	// The discarded vertex is removed from the shape model
	auto V_merge_discard_it = std::find (this -> vertices.begin(),
	                                     this -> vertices.end(),
	                                     V_merge_discard);

	this -> vertices.erase(V_merge_discard_it);



	std::cout << "Done removing discarded vertex" << std::endl;


	unsigned int size_seen_facets_before = seen_facets -> size();

	seen_facets -> erase(F0_old);
	seen_facets -> erase(F1_old);

	if (spurious_facets != nullptr) {
		spurious_facets -> erase(F0_old);
		spurious_facets -> erase(F1_old);
	}



	if (size_seen_facets_before - 2 != seen_facets -> size() && spurious_facets == nullptr) {
		throw (std::runtime_error("facet removal failed: difference = " + std::to_string(size_seen_facets_before - seen_facets -> size())));
	}


	std::cout << "cleaning up" << std::endl;

	// The facets to recycle are removed from the shape model
	auto old_facet_F0 = std::find (this -> facets.begin(), this -> facets.end(), F0_old);
	delete(*old_facet_F0);
	this -> facets.erase(old_facet_F0);

	auto old_facet_F1 = std::find (this -> facets.begin(), this -> facets.end(), F1_old);
	delete(*old_facet_F1);
	this -> facets.erase(old_facet_F1);



	// The impacted facets are all updated to reflect their new geometry
	this -> update_facets(false);

	std::cout << "done" << std::endl;



	// Check if there are any dangling vertex




	if ( V_merge_keep -> get_number_of_owning_facets() < 3 ) {
		throw (std::runtime_error("Dangling vertex leaving merge: V_merge_keep was owned by " + std::to_string(V_merge_keep -> get_number_of_owning_facets()) + " facets"));
	}



	if ( V_keep_0 -> get_number_of_owning_facets() < 3 ) {
		throw (std::runtime_error("Dangling vertex leaving merge: V_keep_0 was owned by " + std::to_string(V_keep_0 -> get_number_of_owning_facets()) + " facets"));
	}



	if ( V_keep_1 -> get_number_of_owning_facets() < 3 ) {
		throw (std::runtime_error("Dangling vertex leaving merge: V_keep_1 was owned by " + std::to_string(V_keep_1 -> get_number_of_owning_facets()) + " facets"));
	}


	if ( V_keep_1 -> common_facets(V_merge_keep).size() != 2) {
		throw (std::runtime_error("V_keep_1 and V_merge_keep share " + std::to_string(V_keep_1 -> common_facets(V_merge_keep).size()) + " facets"));
	};

	if ( V_keep_0 -> common_facets(V_merge_keep).size() != 2) {
		throw (std::runtime_error("V_keep_0 and V_merge_keep share " + std::to_string(V_keep_0 -> common_facets(V_merge_keep).size()) + " facets"));
	};





	return true;

}







void ShapeModel::get_bounding_box(double * bounding_box) const {

	double xmin = std::numeric_limits<double>::infinity();
	double ymin = std::numeric_limits<double>::infinity();
	double zmin = std::numeric_limits<double>::infinity();

	double xmax =  - std::numeric_limits<double>::infinity();
	double ymax =  - std::numeric_limits<double>::infinity();
	double zmax =  - std::numeric_limits<double>::infinity();

	#pragma omp parallel for reduction(max : xmax,ymax,zmax),reduction(min : xmin,ymin,zmin)
	for ( unsigned int vertex_index = 0; vertex_index < this -> get_NVertices(); ++ vertex_index) {

		double * vertex_cords = this -> vertices[vertex_index] -> get_coordinates() -> colptr(0);

		if (vertex_cords[0] >= xmax) {
			xmax = vertex_cords[0];
		}
		else if (vertex_cords[0] <= xmin) {
			xmin = vertex_cords[0];
		}

		if (vertex_cords[1] >= ymax) {
			ymax = vertex_cords[1];
		}
		else if (vertex_cords[1] <= ymin) {
			ymin = vertex_cords[1];
		}

		if (vertex_cords[2] >= zmax) {
			zmax = vertex_cords[2];
		}
		else if (vertex_cords[2] <= zmin) {
			zmin = vertex_cords[2];
		}

	}

	bounding_box[0] = xmin;
	bounding_box[1] = ymin;
	bounding_box[2] = zmin;
	bounding_box[3] = xmax;
	bounding_box[4] = ymax;
	bounding_box[5] = zmax;


	std::cout << "xmin : " << xmin << std::endl;
	std::cout << "xmax : " << xmax << std::endl;


	std::cout << "ymin : " << ymin << std::endl;
	std::cout << "ymax : " << ymax << std::endl;


	std::cout << "zmin : " << zmin << std::endl;
	std::cout << "zmax : " << zmax << std::endl;


}


void ShapeModel::enforce_mesh_quality(double min_facet_angle,
                                      double min_edge_angle,
                                      unsigned int max_recycled_facets,
                                      std::set<Facet * > & seen_facets) {

	bool recycling_still_occuring = true;
	unsigned int facets_recycled = 0;




	while (recycling_still_occuring == true && 2 * facets_recycled < max_recycled_facets) {

		recycling_still_occuring = false;

		for (auto it_facet = seen_facets.begin();
		        it_facet != seen_facets.end();
		        ++ it_facet) {




			if (std::find(this -> facets.begin(), this -> facets.end(), *it_facet) == this -> facets.end()) {
				throw (std::runtime_error("This facet does not exist in the shape model anymore"));
			}


			// This will collapse an edge of the shape model
			// if it appears that the two facets it connects have
			// spurious surface normal orientations


			// if ((*it_facet) -> has_good_edge_quality(min_edge_angle) == false) {

			// };


			if ((*it_facet) -> has_good_surface_quality(min_facet_angle) == false) {

				std::cout << (*it_facet) -> get_facet_center() -> t() << std::endl;
				recycling_still_occuring = this -> merge_shrunk_facet((*it_facet), &seen_facets);

				if (recycling_still_occuring == true) {
					++facets_recycled;

					break;
				}

			}

		}

	}

	std::cout << std::to_string(2 * facets_recycled) << " facets were recycled" << std::endl;

}

void ShapeModel::set_ref_frame_name(std::string ref_frame_name) {

	this -> ref_frame_name = ref_frame_name;
}

