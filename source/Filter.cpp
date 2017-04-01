#include "Filter.hpp"

Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model,
               ShapeModel * estimated_shape_model,
               double t0,
               double tf,
               double omega,
               double min_normal_observation_angle,
               double min_facet_normal_angle_difference,
               unsigned int minimum_ray_per_facet,
               double ridge_coef) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> t0 = t0;
	this -> tf = tf;
	this -> omega = omega;
	this -> min_normal_observation_angle = min_normal_observation_angle;
	this -> min_facet_normal_angle_difference = min_facet_normal_angle_difference;
	this -> minimum_ray_per_facet = minimum_ray_per_facet;
	this -> ridge_coef = ridge_coef;
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

		std::cout << "\n##################################################" << std::endl;
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

	this -> get_observed_features(good_rays,
	                              seen_vertices,
	                              seen_facets,
	                              N_mat,
	                              facet_to_index_of_vertices);

	std::cout << " Number of good rays : " << good_rays.size() << std::endl;
	std::cout << " Facets in view : " << seen_facets.size() << std::endl;

	this -> correct_observed_features(good_rays,
	                                  seen_vertices,
	                                  seen_facets,
	                                  N_mat,
	                                  facet_to_index_of_vertices);

}

void Filter::correct_observed_features(std::vector<Ray * > & good_rays,
                                       std::set<Vertex *> & seen_vertices,
                                       std::set<Facet *> & seen_facets,
                                       arma::mat & N_mat,
                                       std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices) {

	unsigned int gamma = N_mat.n_cols;

	// The information and normal matrices are declared
	arma::mat info_mat = arma::zeros<arma::mat>(gamma, gamma);
	arma::vec normal_mat = arma::zeros<arma::vec>(gamma);
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

		// The information matrix and the normal matrices are augmented
		info_mat += (H_tilde * N_mat).t() * (H_tilde * N_mat);
		normal_mat += (H_tilde * N_mat).t() * ray -> get_range_residual();

		// Htilde is reset
		H_tilde = 0 * H_tilde;

	}

	info_mat = info_mat + this -> ridge_coef * arma::eye<arma::mat>(info_mat.n_rows, info_mat.n_cols);


	// The deviation in the coordinates of the vertices that were seen is computed
	arma::vec alpha = arma::solve(info_mat, normal_mat);
	arma::vec dV = N_mat * alpha;

	std::cout << N_mat << std::endl;
	std::cout << alpha << std::endl;
	std::cout << dV << std::endl;

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
                                   std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices) {




	std::map<Facet *, std::vector<Ray * > > facet_to_rays;
	std::map<Facet *, unsigned int  > hit_count;

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

				u = this -> frame_graph -> convert(u,
				                                   this -> lidar -> get_ref_frame_name(),
				                                   this -> estimated_shape_model -> get_ref_frame_name(),
				                                   true);

				if (std::abs(arma::dot(u, *ray -> get_computed_hit_facet() -> get_facet_normal())) < std::sin(this -> min_normal_observation_angle)) {
					// This is a grazing ray that should be excluded
					continue;
				}

				facet_to_rays[ray -> get_computed_hit_facet()].push_back(ray);

			}
		}
	}


	std::cout << "Facet hit count of the " << facet_to_rays.size() << " facets seen before removing under-observed facets" << std::endl;
	for (auto pair : facet_to_rays) {
		std::cout << pair.second.size() << std::endl;
	}

	for (auto facet_pair : facet_to_rays) {

		if (facet_pair.second.size() >= this -> minimum_ray_per_facet) {

			seen_facets.insert(facet_pair.first);
			seen_vertices.insert(facet_pair.first -> get_vertices() -> at(0) . get());
			seen_vertices.insert(facet_pair.first -> get_vertices() -> at(1) . get());
			seen_vertices.insert(facet_pair.first -> get_vertices() -> at(2) . get());



			for (unsigned int ray_index = 0; ray_index < facet_pair.second.size();
			        ++ray_index) {
				good_rays.push_back(facet_pair.second[ray_index]);
				hit_count[facet_pair.first] += 1;

			}

		}

	}



	std::cout << "Facet hit count of the " << hit_count.size() << " facets seen after removing under-observed facets" << std::endl;
	for (auto pair : hit_count) {
		std::cout << pair.second << std::endl;
	}




	// This will help counting how many facets each vertex belongs to
	// std::map<unsigned int, std::vector<unsigned int> > vertex_to_owning_facets;
	std::map<unsigned int, std::vector<Facet *> > vertex_to_owning_facets;

	for (unsigned int seen_facet_index = 0;
	        seen_facet_index < seen_facets . size();
	        ++seen_facet_index) {

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

		vertex_to_owning_facets[v0_index].push_back(facet);
		vertex_to_owning_facets[v1_index].push_back(facet);
		vertex_to_owning_facets[v2_index].push_back(facet);

		facet_to_index_of_vertices[facet].push_back(v0_index);
		facet_to_index_of_vertices[facet].push_back(v1_index);
		facet_to_index_of_vertices[facet].push_back(v2_index);

	}


	// Can only solve for displacements
	// along independent directions
	std::map<unsigned int, std::vector<arma::vec *> > vertex_to_normal;

	for (unsigned int v_index = 0; v_index < seen_vertices.size(); ++ v_index) {

		// Easy: that vertex can only move along one direction
		if (vertex_to_owning_facets[v_index].size() == 1) {
			vertex_to_normal[v_index].push_back(vertex_to_owning_facets[v_index][0] -> get_facet_normal());
		}

		// Easy: that vertex is owned by two facets and can mode to
		// up to 2 independent directions
		else if (vertex_to_owning_facets[v_index].size() == 2) {
			arma::vec * n1 = vertex_to_owning_facets[v_index][0] -> get_facet_normal();
			arma::vec * n2 = vertex_to_owning_facets[v_index][1] -> get_facet_normal();

			if (arma::norm(arma::cross(*n1, *n2)) > std::sin(this -> min_facet_normal_angle_difference)) {
				vertex_to_normal[v_index].push_back(n1);
				vertex_to_normal[v_index].push_back(n2);
			}
			else {
				vertex_to_normal[v_index].push_back(n2);
			}

		}

		// This vertex is owned by three facets or more. Have to determine
		// a minimum set of normal to those facets spanning R3.
		else {

			arma::vec * n1;
			arma::vec * n2;
			arma::vec * n3;


			// First of all, two non-colinear facet normals are found
			// The first normal is used as a reference
			// It will be used no matter what
			n1 = vertex_to_owning_facets[v_index][0] -> get_facet_normal();
			vertex_to_normal[v_index].push_back(n1);

			unsigned int n2_index = 0;

			for (unsigned int facet_index = 1; facet_index < vertex_to_owning_facets[v_index].size();
			        ++facet_index) {

				n2 = vertex_to_owning_facets[v_index][facet_index] -> get_facet_normal();

				// If true, we just found another independent normal.
				// We need to select up to one more
				if (arma::norm(arma::cross(*n1, *n2)) > std::sin(this -> min_facet_normal_angle_difference)) {
					vertex_to_normal[v_index].push_back(n2);
					n2_index = facet_index;
					break;
				}

			}

			// If all the facet normals were not colinear, then at least two were selected and we can
			// try to look for a third one
			if (n2_index != 0) {

				for (unsigned int facet_index = 1; facet_index < vertex_to_owning_facets[v_index].size();
				        ++facet_index) {

					if (facet_index != n2_index) {

						n3 = vertex_to_owning_facets[v_index][facet_index] -> get_facet_normal();
						// If true, we found our third normal
						if (std::abs(arma::dot(arma::cross(*n1, *n2), *n3)) > std::sin(this -> min_facet_normal_angle_difference)) {
							vertex_to_normal[v_index].push_back(n3);
							break;
						}

					}
				}
			}



		}

	}


	unsigned int gamma = 0;
	for (unsigned int seen_vertex_index = 0; seen_vertex_index < seen_vertices.size(); ++ seen_vertex_index) {
		gamma += vertex_to_normal[seen_vertex_index].size();
	}

	// The matrix mapping the normal displacement of each facet to that of the vertices
	// if constructed
	N_mat = arma::zeros<arma::mat>(3 * seen_vertices.size(), gamma);

	unsigned int col_index = 0;
	for (unsigned int v_index = 0; v_index < seen_vertices.size(); ++ v_index) {

		for (unsigned int normal_index_local = 0; normal_index_local < vertex_to_normal[v_index].size(); ++normal_index_local) {

			N_mat.rows(3 * v_index, 3 * v_index + 2).col(col_index) = *vertex_to_normal[v_index][normal_index_local];
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

