#include "Filter.hpp"

Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model,
               ShapeModel * estimated_shape_model,
               double t0,
               double tf,
               double omega,
               double theta_min) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> t0 = t0;
	this -> tf = tf;
	this -> omega = omega;
	this -> theta_min = theta_min;
}


void Filter::run(unsigned int N_iteration, bool plot_measurements, bool save_shape_model) {

	std::cout << "Running the filter" << std::endl;

	// The vector of times is created first
	// It corresponds to the observation times
	std::vector<double> times;

	times.push_back(this -> t0);
	double t = times[0];

	while (t < this -> tf) {
		t = t + 1. / this -> lidar -> get_frequency();
		times.push_back(t);
	}


	arma::vec lidar_pos_0 = *(this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> get_origin_from_parent());
	arma::vec lidar_pos = arma::vec(3);

	arma::vec u =  arma::vec(3);
	arma::vec v =  arma::vec(3);
	arma::vec w =  {0, 0, 1};
	arma::mat dcm_LN = arma::zeros<arma::mat>(3, 3);
	arma::vec mrp_LN = arma::vec(3);
	dcm_LN.row(2) = w.t();



	if (save_shape_model == true) {
		this -> estimated_shape_model -> save("../output/shape_model/shape_model_apriori.obj");
		this -> true_shape_model -> save("../output/shape_model/true_shape_model.obj");

	}


	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {

		// The lidar is on a circular trajectory and is manually steered to the
		// asteroid

		lidar_pos = M3(-this -> omega * times[time_index]) * lidar_pos_0;
		u = arma::normalise( - lidar_pos);
		v = arma::normalise(arma::cross(w, u));

		std::cout << time_index << " " << 180. / arma::datum::pi * this -> 	omega * times[time_index] << std::endl;



		dcm_LN.row(0) = u.t();
		dcm_LN.row(1) = v.t();
		mrp_LN = dcm_to_mrp(dcm_LN);



		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		this -> lidar -> send_flash(this -> true_shape_model, false);

		if (plot_measurements == true) {
			this -> lidar -> send_flash(this -> estimated_shape_model, true);
			this -> lidar -> plot_ranges("../output/measurements/residuals_prefit_" + std::to_string(time_index), 2);
			this -> lidar -> plot_ranges("../output/measurements/computed_prefit_" + std::to_string(time_index), 1);
			this -> lidar -> plot_ranges("../output/measurements/true_" + std::to_string(time_index), 0);

		}



		// The filter is iterated N times
		for (unsigned int iteration = 0; iteration < N_iteration ; ++iteration) {
			this -> lidar -> send_flash(this -> estimated_shape_model, true);
			this -> correct_shape();
		}

		// The postfit residuals are stored
		if (plot_measurements == true) {
			this -> lidar -> send_flash(this -> estimated_shape_model, true);
			this -> lidar -> plot_ranges("../output/measurements/residuals_postfit_" + std::to_string(time_index), 2);
			this -> lidar -> plot_ranges("../output/measurements/computed_postfit_" + std::to_string(time_index), 1);
		}


		if (save_shape_model == true) {
			this -> estimated_shape_model -> save("../output/shape_model/shape_model_" + std::to_string(time_index) + ".obj");
		}

	}


}

void Filter::correct_shape() {

	std::vector<Ray * > good_rays;
	std::set<Vertex *> seen_vertices;
	std::set<Facet *> seen_facets;
	arma::mat N_mat;
	std::map<Facet *, std::vector<unsigned int> > facet_to_index_of_vertices;
	std::map<Facet *, arma::uvec> facet_to_N_mat_cols;

	this -> get_observed_features(good_rays, seen_vertices,
	                              seen_facets,
	                              N_mat,
	                              facet_to_index_of_vertices,
	                              facet_to_N_mat_cols);
	// std::cout << " Facets in view : " << seen_facets.size() << std::endl;
	this -> correct_observed_features(good_rays,
	                                  seen_vertices,
	                                  seen_facets,
	                                  N_mat,
	                                  facet_to_index_of_vertices,
	                                  facet_to_N_mat_cols);

}

void Filter::correct_observed_features(std::vector<Ray * > & good_rays,
                                       std::set<Vertex *> & seen_vertices,
                                       std::set<Facet *> & seen_facets,
                                       arma::mat & N_mat,
                                       std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices,
                                       std::map<Facet *, arma::uvec> & facet_to_N_mat_cols) {

	unsigned int gamma = N_mat.n_cols;

	// The information and normal matrices are declared
	arma::mat info_mat = arma::zeros<arma::mat>(
	                         gamma,
	                         gamma);

	arma::vec normal_mat = arma::zeros<arma::vec>(
	                           gamma);


	arma::rowvec H_tilde = arma::zeros<arma::rowvec>( 3 * seen_vertices . size());

	for (unsigned int ray_index = 0; ray_index < good_rays.size(); ++ray_index) {

		Ray * ray = good_rays . at(ray_index);

		// The origin and direction of the ray are converted from the
		// lidar frame to the estimated shape model frame
		arma::vec u = *(ray -> get_direction());
		arma::vec P = *(ray -> get_origin());

		u = this -> frame_graph -> convert(
		        u,
		        this -> lidar -> get_ref_frame_name(),
		        this -> estimated_shape_model -> get_ref_frame_name(),
		        true);

		P = this -> frame_graph -> convert(
		        P,
		        this -> lidar -> get_ref_frame_name(),
		        this -> estimated_shape_model -> get_ref_frame_name(),
		        false);



		// The partial derivatives are computed
		std::vector<arma::rowvec> partials = this -> partial_range_partial_coordinates(P,
		                                     u,
		                                     ray -> get_computed_hit_facet());


		// The indices of the facet present in the facet
		// impacted by this ray are found
		// Here, the indices are purely local
		// to the dimension of seen_vertices

		unsigned int v0_index = facet_to_index_of_vertices[ ray -> get_computed_hit_facet()][0];
		unsigned int v1_index = facet_to_index_of_vertices[ ray -> get_computed_hit_facet()][1];
		unsigned int v2_index = facet_to_index_of_vertices[ ray -> get_computed_hit_facet()][2];

		// The corresponding partitions of H_tilde are set
		H_tilde.cols(3 * v0_index, 3 * v0_index + 2) = partials[0];
		H_tilde.cols(3 * v1_index, 3 * v1_index + 2) = partials[1];
		H_tilde.cols(3 * v2_index, 3 * v2_index + 2) = partials[2];

		// A temporary N_facet is necessary.
		// Using N would put responsibility on
		// all the facets owning an edge shared by the facet
		// hit by the ray
		// arma::mat N_facet = arma::zeros<arma::mat>(N_mat.n_rows, N_mat.n_cols);

		// The columns of N/N_facet corresponding to a normal vector different
		// from the one of the facet hit by the ray are set to zero
		// N_mat.cols(facet_to_N_mat_cols[ray -> get_computed_hit_facet()]) = N_mat.cols(facet_to_N_mat_cols[ray -> get_computed_hit_facet()]);

		// The information matrix and the normal matrices are augmented
		info_mat += (H_tilde * N_mat).t() * (H_tilde * N_mat);
		normal_mat += (H_tilde * N_mat).t() * ray -> get_range_residual();


		// Htilde is reset
		H_tilde = 0 * H_tilde;

	}

	// The deviation in the coordinates of the vertices that were seen is computed
	arma::vec alpha = arma::solve(info_mat, normal_mat);
	arma::vec dV = N_mat * alpha;
	// std::cout << N_final << std::endl;
	// std::cout << alpha << std::endl;
	// std::cout << dV << std::endl;

	// The location of the vertices is updated
	for (unsigned int vertex_index = 0; vertex_index < seen_vertices.size(); ++vertex_index) {

		Vertex * vertex = *std::next(seen_vertices.begin(), vertex_index);
		*vertex-> get_coordinates() = *vertex -> get_coordinates() +
		                              dV.rows(3 * vertex_index , 3 * vertex_index + 2 );

	}

	// The mass properties of the shape model are recomputed
	// (center of mass, volume, surface area)
	this -> estimated_shape_model -> update_mass_properties();

	// The shape model is shifted using the new location of the center of mass
	// this -> estimated_shape_model -> shift(-(*this -> estimated_shape_model -> get_center_of_mass()));

	// The facets of the shape model that have been seen are updated
	// There is some overhead because their center will be recomputed
	// after having been shifted, but it's not too bad since
	// seen_facets is only a fraction of the estimated shape model facets
	this -> estimated_shape_model -> update_facets();

	// std::cout << "Volume:" << this -> estimated_shape_model -> get_volume() << std::endl;

}

void Filter::get_observed_features(std::vector<Ray * > & good_rays,
                                   std::set<Vertex *> & seen_vertices,
                                   std::set<Facet *> & seen_facets,
                                   arma::mat & N_mat,
                                   std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices,
                                   std::map<Facet *, arma::uvec> & facet_to_N_mat_cols) {

	for (unsigned int row_index = 0; row_index < this -> lidar -> get_row_count(); ++row_index) {

		for (unsigned int col_index = 0; col_index < this -> lidar  -> get_col_count(); ++col_index) {
			Ray * ray = this -> lidar -> get_ray(row_index, col_index);

			// If either the true target or the a-priori
			// shape were missed, this measurement is
			// unusable
			if (ray -> get_computed_hit_facet() == nullptr
			        || ray -> get_true_hit_facet() == nullptr) {
				continue;
			}
			else {

				// Grazing rays are excluded
				arma::vec u = *(ray -> get_direction());

				u = this -> frame_graph -> convert(
				        u,
				        this -> lidar -> get_ref_frame_name(),
				        this -> estimated_shape_model -> get_ref_frame_name(),
				        true);
				if (std::abs(arma::dot(u, *ray -> get_computed_hit_facet() -> get_facet_normal())) < std::sin(this -> theta_min)) {
					// This is a grazing ray that should be excluded
					continue;
				}

				good_rays. push_back(ray);
				seen_facets. insert(ray -> get_computed_hit_facet());
				seen_vertices. insert(ray -> get_computed_hit_facet() -> get_vertices() -> at(0) . get());
				seen_vertices. insert(ray -> get_computed_hit_facet() -> get_vertices() -> at(1) . get());
				seen_vertices. insert(ray -> get_computed_hit_facet() -> get_vertices() -> at(2) . get());

			}

		}
	}

	// This will help counting how many facets each vertex belongs to
	std::map<unsigned int, std::vector<unsigned int> > vertex_to_owning_facets;

	for (unsigned int seen_vertex_index = 0; seen_vertex_index < seen_vertices.size(); ++ seen_vertex_index) {
		std::vector<unsigned int> owning_facet_indices;
		vertex_to_owning_facets[seen_vertex_index] = owning_facet_indices;
	}

	for (unsigned int seen_facet_index = 0; seen_facet_index < seen_facets . size(); ++seen_facet_index) {

		Facet * facet = *std::next(seen_facets . begin(), seen_facet_index);

		unsigned int v0_index = std::distance(
		                            seen_vertices . begin(),
		                            seen_vertices . find(
		                                facet -> get_vertices() -> at(0).get())
		                        );

		unsigned int v1_index = std::distance(
		                            seen_vertices . begin(),
		                            seen_vertices . find(
		                                facet -> get_vertices() -> at(1).get())
		                        );


		unsigned int v2_index = std::distance(
		                            seen_vertices . begin(),
		                            seen_vertices . find(
		                                facet -> get_vertices() -> at(2).get())
		                        );

		vertex_to_owning_facets[v0_index].push_back(seen_facet_index);
		vertex_to_owning_facets[v1_index].push_back(seen_facet_index);
		vertex_to_owning_facets[v2_index].push_back(seen_facet_index);

		facet_to_index_of_vertices[facet].push_back(v0_index);
		facet_to_index_of_vertices[facet].push_back(v1_index);
		facet_to_index_of_vertices[facet].push_back(v2_index);

	}

	unsigned int gamma = 0;
	for (unsigned int seen_vertex_index = 0; seen_vertex_index < seen_vertices.size(); ++ seen_vertex_index) {
		gamma += vertex_to_owning_facets[seen_vertex_index].size();
	}

	// The matrix mapping the normal displacement of each facet to that of the vertices
	// if constructed
	N_mat = arma::zeros<arma::mat>(3 * seen_vertices.size(), gamma);

	// needed in order to use fixed size uvec afterwards
	std::map<Facet *, std::vector<unsigned int> > facet_to_N_mat_cols_temp;



	unsigned int col_index = 0;

	for (unsigned int v_index = 0; v_index < seen_vertices.size(); ++ v_index) {

		for (unsigned int facet_index_local = 0; facet_index_local < vertex_to_owning_facets[v_index].size(); ++facet_index_local) {

			unsigned int facet_index = vertex_to_owning_facets[v_index][facet_index_local];
			Facet * facet = *std::next(seen_facets . begin(), facet_index);

			facet_to_N_mat_cols_temp[facet].push_back(col_index);

			N_mat.rows(3 * v_index, 3 * v_index + 2).col(col_index) = *facet -> get_facet_normal() ;
			++col_index;
		}
	}


}


std::vector<arma::rowvec> Filter::partial_range_partial_coordinates(const arma::vec & P, const arma::vec & u, Facet * facet) {

	std::vector<arma::rowvec> partials;

	// It is required to "de-normalized" the normal
	// vector so as to have a consistent
	// partial derivative
	arma::vec n = 2 * facet -> get_area() * (*facet -> get_facet_normal());

	std::vector<std::shared_ptr<Vertex > > * vertices = facet -> get_vertices();

	arma::vec * V0 =  vertices -> at(0) -> get_coordinates();
	arma::vec * V1 =  vertices -> at(1) -> get_coordinates();
	arma::vec * V2 =  vertices -> at(2) -> get_coordinates();



	arma::rowvec drhodV0 = (n.t()) / arma::dot(u, n) + (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * tilde(*V2 - *V1);


	arma::rowvec drhodV1 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * tilde(*V0 - *V2);


	arma::rowvec drhodV2 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * tilde(*V1 - *V0);

	partials.push_back(drhodV0);
	partials.push_back(drhodV1);
	partials.push_back(drhodV2);

	return partials;





}

