#include "ShapeModelImporter.hpp"

ShapeModelImporter::ShapeModelImporter(std::string filename, double scaling_factor, bool as_is) {
	this -> filename = filename;
	this -> scaling_factor = scaling_factor;
	this -> as_is = as_is;
}

void ShapeModelImporter::load_shape_model(ShapeModelTri * shape_model) const {

	std::ifstream ifs(this -> filename);

	if (!ifs.is_open()) {
		std::cout << "There was a problem opening the input file!\n";
		throw;
	}

	std::string line;
	std::vector<arma::vec> vertices;
	std::vector<arma::uvec> facet_vertices;

	std::cout << " Reading " << this -> filename << std::endl;

	while (std::getline(ifs, line)) {

		std::stringstream linestream(line);

		std::string word1, word2, word3;

		char type;
		linestream >> type;

		if (type == '#' || type == 's'  || type == 'o' || type == 'm' || type == 'u' || line.size() == 0) {
			continue;
		}

		else if (type == 'v') {
			double vx, vy, vz;
			linestream >> vx >> vy >> vz;
			arma::vec vertex = {vx, vy, vz};
			vertices.push_back(this -> scaling_factor * vertex);

		}

		else if (type == 'f') {
			unsigned int v0, v1, v2;
			linestream >> v0 >> v1 >> v2;

			arma::uvec vertices_in_facet = {v0 - 1, v1 - 1, v2 - 1};
			facet_vertices.push_back(vertices_in_facet);

		}

		else {
			std::cout << " unrecognized type: " << type << std::endl;
			throw;
		}

	}

	std::cout << " Number of vertices: " << vertices.size() << std::endl;
	std::cout << " Number of facets: " << facet_vertices.size() << std::endl;


	// Vertices are added to the shape model
	std::vector<std::shared_ptr<ControlPoint>> vertex_index_to_ptr(vertices.size(), nullptr);

	std::cout << std::endl << " Constructing Vertices " << std::endl  ;
	boost::progress_display progress_vertices(vertices.size()) ;

	for (unsigned int vertex_index = 0; vertex_index < vertices.size(); ++vertex_index) {

		std::shared_ptr<arma::vec> coordinates = std::make_shared<arma::vec>(vertices[vertex_index]);

		std::shared_ptr<ControlPoint> vertex = std::make_shared<ControlPoint>(ControlPoint());
		vertex -> set_coordinates(coordinates);

		vertex_index_to_ptr[vertex_index] = vertex;
		shape_model -> add_vertex(vertex);
		++progress_vertices;

	}

	std::cout << std::endl << " Constructing Facets " << std::endl ;


	boost::progress_display progress_facets(facet_vertices.size()) ;

	// Facets are added to the shape model
	for (unsigned int facet_index = 0; facet_index < facet_vertices.size(); ++facet_index) {

		// The vertices stored in this facet are pulled.
		std::shared_ptr<ControlPoint> v0 = vertex_index_to_ptr[facet_vertices[facet_index][0]];
		std::shared_ptr<ControlPoint> v1 = vertex_index_to_ptr[facet_vertices[facet_index][1]];
		std::shared_ptr<ControlPoint> v2 = vertex_index_to_ptr[facet_vertices[facet_index][2]];

		std::vector<std::shared_ptr<ControlPoint>> vertices;
		vertices.push_back(v0);
		vertices.push_back(v1);
		vertices.push_back(v2);


		
		std::shared_ptr<Facet> facet = std::make_shared<Facet>(Facet(std::vector<std::shared_ptr<ControlPoint>>(vertices)));

		shape_model -> add_facet(facet);
		++progress_facets;
	}

	// The surface area, volume, center of mass of the shape model
	// are computed
	shape_model -> update_mass_properties();

	if (this -> as_is == false) {

		// The shape model is shifted so as to have its coordinates
		// expressed in its barycentric frame
		shape_model -> shift_to_barycenter();

		// The shape model is then rotated so as to be oriented
		// with respect to its principal axes
		// The inertia tensor is computed on this occasion
		shape_model -> align_with_principal_axes();

	}

	// Facets are updated (their normals and centers
	// are computed) to reflect the new position/orientation
	shape_model -> update_facets();


	// The consistency of the surface normals is checked
	shape_model -> check_normals_consistency();


}
