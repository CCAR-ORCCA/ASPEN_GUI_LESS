#include "../include/ShapeModel.hpp"
#include <chrono>

// void ShapeModel::load(const std::string & filename) {

// 	Assimp::Importer importer;

// 	const aiScene * scene = importer.ReadFile( filename, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices  );

// 	// If the import succeeded:
// 	if (scene != NULL) {

// 		// If the imported shape model had at least one mesh in it
// 		if (scene -> mMeshes > 0) {

// 			// For now, only the first mesh is used
// 			this -> vertices = arma::mat(3, scene -> mMeshes[0] -> mNumVertices);
// 			this -> facet_vertices = arma::umat(3, scene -> mMeshes[0] -> mNumFaces);
// 			this -> facet_normals = arma::mat(3, scene -> mMeshes[0] -> mNumFaces);
// 			this -> F_dyads = arma::cube(scene -> mMeshes[0] -> mNumFaces, 3, 3);

// 			this -> NFacets = scene -> mMeshes[0] -> mNumFaces;
// 			this -> NVertices = scene -> mMeshes[0] -> mNumVertices;

// 			std::map<unsigned int , std::set<unsigned int> > vertex_index_to_facet;
// 			std::set<std::set<unsigned int> > edges;

// 			// Vertex coordinates



// 			for (unsigned int vertex = 0; vertex < scene -> mMeshes[0] -> mNumVertices; ++vertex) {

// 				arma::vec vertex_coords = {scene -> mMeshes[0] -> mVertices[vertex].x,
// 				                           scene -> mMeshes[0] -> mVertices[vertex].y,
// 				                           scene -> mMeshes[0] -> mVertices[vertex].z
// 				                          };
// 				this -> vertices.col(vertex) = vertex_coords;
// 			}


// 			// Connectivity Table
// 			for (unsigned int facet = 0; facet < this -> NFacets ; ++facet) {

// 				if (scene -> mMeshes[0] -> mFaces[facet].mNumIndices != 3) {
// 					std::cout << " More than three vertices belong to this facet " << std::endl;
// 					throw " More than three vertices belong to this facet ";
// 				}

// 				this -> facet_vertices.col(facet)(0) = scene -> mMeshes[0] -> mFaces[facet].mIndices[0];
// 				this -> facet_vertices.col(facet)(1) = scene -> mMeshes[0] -> mFaces[facet].mIndices[1];
// 				this -> facet_vertices.col(facet)(2) = scene -> mMeshes[0] -> mFaces[facet].mIndices[2];


// 				vertex_index_to_facet[this -> facet_vertices.col(facet)(0)].insert(facet);
// 				vertex_index_to_facet[this -> facet_vertices.col(facet)(1)].insert(facet);
// 				vertex_index_to_facet[this -> facet_vertices.col(facet)(2)].insert(facet);

// 				std::set<unsigned int> edge_0;
// 				edge_0.insert(scene -> mMeshes[0] -> mFaces[facet].mIndices[0]);
// 				edge_0.insert(scene -> mMeshes[0] -> mFaces[facet].mIndices[1]);

// 				std::set<unsigned int> edge_1;
// 				edge_1.insert(scene -> mMeshes[0] -> mFaces[facet].mIndices[0]);
// 				edge_1.insert(scene -> mMeshes[0] -> mFaces[facet].mIndices[2]);

// 				std::set<unsigned int> edge_2;
// 				edge_2.insert(scene -> mMeshes[0] -> mFaces[facet].mIndices[1]);
// 				edge_2.insert(scene -> mMeshes[0] -> mFaces[facet].mIndices[2]);

// 				if (this -> edges_to_facets.find(edge_0) == this -> edges_to_facets.end()) {
// 					this -> edges_to_facets[edge_0].insert(facet);
// 				}

// 				else if (this -> edges_to_facets[edge_0].size() < 2) {
// 					this -> edges_to_facets[edge_0].insert(facet);
// 				}

// 				if (this -> edges_to_facets.find(edge_1) == this -> edges_to_facets.end()) {
// 					this -> edges_to_facets[edge_1].insert(facet);
// 				}

// 				else if (this -> edges_to_facets[edge_1].size() < 2) {
// 					this -> edges_to_facets[edge_1].insert(facet);
// 				}


// 				if (this -> edges_to_facets.find(edge_2) == this -> edges_to_facets.end()) {
// 					this -> edges_to_facets[edge_2].insert(facet);
// 				}

// 				else if (this -> edges_to_facets[edge_2].size() < 2) {
// 					this -> edges_to_facets[edge_2].insert(facet);
// 				}

// 				edges.insert(edge_0);
// 				edges.insert(edge_1);
// 				edges.insert(edge_2);

// 			}

// 			this -> NEdges = edges.size();
// 			this -> E_dyads = arma::cube(this -> NEdges, 3, 3);
// 			unsigned int edge_index = 0;


// 			for (std::set<std::set<unsigned int> >::iterator iter = edges.begin(); iter != edges.end(); ++iter) {
// 				this -> edges_to_edges_index[*iter] = edge_index;
// 				this -> edges_indices_to_edge[edge_index] = *iter;
// 				++edge_index;
// 			}

// 			// Normals
// 			#pragma omp parallel for
// 			for (unsigned int facet = 0; facet < this -> NFacets; ++facet) {
// 				unsigned int P0_index = this -> facet_vertices.col(facet)(0);
// 				unsigned int P1_index = this -> facet_vertices.col(facet)(1);
// 				unsigned int P2_index = this -> facet_vertices.col(facet)(2);

// 				arma::vec P0 = this -> vertices.col(P0_index);
// 				arma::vec P1 = this -> vertices.col(P1_index);
// 				arma::vec P2 = this -> vertices.col(P2_index);
// 				arma::vec facet_normal = arma::cross(P1 - P0, P2 - P0) / arma::norm(arma::cross(P1 - P0, P2 - P0));
// 				this -> facet_normals.col(facet) = facet_normal;
// 			}m


// 			this -> check_normals_consistency();

// 			this -> compute_dyads();

// 		}

// 	}

// 	else {
// 		std::cout << " There was an error opening the shape model file " << std::endl;
// 		throw " There was an error opening the shape model file ";
// 	}

// }

ShapeModel::ShapeModel() {

}

void ShapeModel::update_mass_properties() {
	this -> compute_surface_area();

	this -> compute_volume();

	this -> compute_center_of_mass();

}


void ShapeModel::update_facets() {

	for (auto & facet : this -> facets) {
		facet -> update();
	}

}

void ShapeModel::update_facets(std::set<Facet *> & facets) {

	for (auto & facet : facets) {
		facet -> update();
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


void ShapeModel::shift(arma::vec x) {

	// The vertices are shifted
	#pragma omp parallel for if(USE_OMP_SHAPE_MODEL)
	for (unsigned int vertex_index = 0;
	        vertex_index < this -> get_NVertices();
	        ++vertex_index) {

		*this -> vertices[vertex_index] -> get_coordinates() = *this -> vertices[vertex_index] -> get_coordinates() + x;

	}

	// The facet centers are shifted
	// Faster than calling update() on the facet as this would also recompute
	// the normals, the dyads and the surface area
	#pragma omp parallel for if(USE_OMP_SHAPE_MODEL)
	for (unsigned int facet_index = 0;
	        facet_index < this -> get_NFacets();
	        ++facet_index) {

		*this -> facets[facet_index] -> get_facet_center() =  *this -> facets[facet_index] -> get_facet_center() + x;
	}


}

ShapeModel::ShapeModel(std::string ref_frame_name,
                       FrameGraph * frame_graph) {
	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
}



void ShapeModel::add_facet(Facet * facet) {
	this -> facets. push_back(facet);
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
		throw (std::runtime_error("Normals were incorrectly oriented. norm(sum(n * s))/sum(s):" + std::to_string(arma::norm(surface_sum) / facet_area_average)));
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


void ShapeModel::split_facet(Facet * facet) {

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

	F0 -> set_split_counter(facet -> get_split_counter() + 1);
	F1 -> set_split_counter(facet -> get_split_counter() + 1);
	F2 -> set_split_counter(facet -> get_split_counter() + 1);
	F3 -> set_split_counter(facet -> get_split_counter() + 1);
	F4 -> set_split_counter(facet -> get_split_counter() + 1);
	F5 -> set_split_counter(facet -> get_split_counter() + 1);
	F6 -> set_split_counter(facet -> get_split_counter() + 1);
	F7 -> set_split_counter(facet -> get_split_counter() + 1);
	F8 -> set_split_counter(facet -> get_split_counter() + 1);
	F9 -> set_split_counter(facet -> get_split_counter() + 1);


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

	V3 -> remove_facet_ownership(F3_old);
	V4 -> remove_facet_ownership(F1_old);
	V5 -> remove_facet_ownership(F2_old);



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

}



void ShapeModel::recycle_facet(Facet * facet) {

	// The vertices in the facet are extracted
	std::shared_ptr<Vertex> V0  = facet -> get_vertices() -> at(0);
	std::shared_ptr<Vertex> V1  = facet -> get_vertices() -> at(1);
	std::shared_ptr<Vertex> V2  = facet -> get_vertices() -> at(2);

	arma::vec * P0  = V0 -> get_coordinates();
	arma::vec * P1  = V1 -> get_coordinates();
	arma::vec * P2  = V2 -> get_coordinates();


	// The smallest of the three angles in the facet is identified
	arma::vec sin_angles = arma::vec(3);
	sin_angles(0) = arma::norm(arma::cross(*P1 - *P0, *P2 - *P0) / ( arma::norm(*P1 - *P0) * arma::norm(*P2 - *P0) ));
	sin_angles(1) = arma::norm(arma::cross(*P2 - *P1, *P0 - *P1) / ( arma::norm(*P2 - *P1) * arma::norm(*P0 - *P1) ));
	sin_angles(2) = arma::norm(arma::cross(*P0 - *P2, *P1 - *P2) / ( arma::norm(*P0 - *P2) * arma::norm(*P1 - *P2) ));


	// This index indicates which vertices are to be merged
	// - 0 : V1 and V2 shoud be merged
	// - 1 : V0 and V2
	// - 2 : V1 and V0
	unsigned int min_angle_index = sin_angles.index_min();


	std::shared_ptr<Vertex> V_merge_keep;
	std::shared_ptr<Vertex> V_merge_discard;
	std::shared_ptr<Vertex> V_keep_0;
	std::shared_ptr<Vertex> V_keep_1;


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


	std::vector<Facet * > facets_owning_discarded_vertex = V_merge_discard -> get_owning_facets();

	// The facets owning V_merge_discard are
	// updated so as to have this vertex merging with
	// V_merge_keep

	for (unsigned int facet_index = 0; facet_index < facets_owning_discarded_vertex.size(); ++facet_index) {

		// This will also update the facets that will be recycled but this is not a big deal
		Facet * facet_to_update = facets_owning_discarded_vertex[facet_index];

		for (unsigned int vertex_index = 0; vertex_index < 3; ++vertex_index) {

			if (facet_to_update -> get_vertices() -> at(vertex_index) == V_merge_discard) {
				facet_to_update -> get_vertices() -> at(vertex_index) = V_merge_keep;
				break;
			}

		}
	}

	// V_keep_0,1 are still owned by the facets to be recycled
	V_keep_0 -> remove_facet_ownership(F0_old);
	V_keep_1 -> remove_facet_ownership(F1_old);

	// The discarded vertex is removed from the shape model
	auto V_merge_discard_it = std::find (this -> vertices.begin(), this -> vertices.end(), V_merge_discard);

	this -> vertices.erase(V_merge_discard_it);

	// At this point, the ref_count of V_merge_discard should yield 1
	// std::cout << " V_merge_discard count: " << V_merge_discard.use_count() << std::endl;
	// Actually it does not because of the edges!

	// The facets to recycle are removed from the shape model
	auto old_facet_F0 = std::find (this -> facets.begin(), this -> facets.end(), facets_to_recycle[0]);
	delete(*old_facet_F0);
	this -> facets.erase(old_facet_F0);

	auto old_facet_F1 = std::find (this -> facets.begin(), this -> facets.end(), facets_to_recycle[1]);
	delete(*old_facet_F1);
	this -> facets.erase(old_facet_F1);

	// The impacted facets are all updated to reflect their new geometry
	this -> update_facets();


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

}


void ShapeModel::enforce_mesh_quality() {

	bool mesh_quality_confirmed = false;
	unsigned int facets_recycled = 0;

	while (mesh_quality_confirmed == false) {

		mesh_quality_confirmed = true;

		for (unsigned int facet_index = 0;
		        facet_index < this -> facets.size();
		        ++facet_index) {

			if (this -> facets[facet_index] -> has_good_quality() == false) {
				std::cout << "Recycling" << std::endl;
				mesh_quality_confirmed = false;
				this -> recycle_facet(this -> facets[facet_index]);
				++facets_recycled;
				break;

			}

		}

	}

	std::cout << std::to_string(2 * facets_recycled) << " facets were recycled" << std::endl;

}

void ShapeModel::set_ref_frame_name(std::string ref_frame_name) {

	this -> ref_frame_name = ref_frame_name;
}

