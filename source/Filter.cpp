#include "Filter.hpp"

Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model,
               ShapeModel * estimated_shape_model,
               double t0,
               double tf,
               double dt) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> t0 = t0;
	this -> tf = tf;
	this -> dt = dt;
}


void Filter::run() {

	// The vector of times is created first
	// It corresponds to the observation times
	std::vector<double> times;

	times.push_back(this -> t0);
	double t = times[0];

	while (t <= this -> tf) {
		t = t + 1. / this -> lidar -> get_frequency();
		times.push_back(t);
	}

	double omega = 1e-2;

	arma::vec lidar_pos_0 = *(this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> get_origin_from_parent());
	arma::vec lidar_pos = arma::vec(3);

	arma::vec u =  arma::vec(3);
	arma::vec v =  arma::vec(3);
	arma::vec w =  {0, 0, 1};
	arma::mat dcm_LN = arma::zeros<arma::mat>(3, 3);
	arma::vec mrp_LN = arma::vec(3);
	dcm_LN.row(2) = w.t();

	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {

		// The lidar is on a circular trajectory and is manually steered to the
		// asteroid

		lidar_pos = M3(-omega * times[time_index]) * lidar_pos_0;
		u = arma::normalise( - lidar_pos);
		v = arma::normalise(arma::cross(w, u));

		dcm_LN.row(0) = u.t();
		dcm_LN.row(1) = v.t();
		mrp_LN = dcm_to_mrp(dcm_LN);

		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		this -> lidar -> send_flash(this -> true_shape_model, false);
		this -> lidar -> send_flash(this -> estimated_shape_model, true);

		this -> lidar -> plot_ranges("../images/true_" + std::to_string(time_index), 0);
		this -> lidar -> plot_ranges("../images/computed_" + std::to_string(time_index), 1);
		this -> lidar -> plot_ranges("../images/residuals_" + std::to_string(time_index), 2);


		this -> correct_shape();

	}


}

void Filter::correct_shape() {

	// The information and normal matrices are declared
	arma::mat info_mat = arma::zeros<arma::mat>(
	                         3 * this -> estimated_shape_model -> get_NVertices(),
	                         3 * this -> estimated_shape_model -> get_NVertices());

	arma::vec normal_mat = arma::zeros<arma::vec>(
	                           3 * this -> estimated_shape_model -> get_NVertices());

	arma::rowvec H_tilde = arma::zeros<arma::rowvec>(3 * this -> estimated_shape_model -> get_NVertices());


	// The measurements are processed
	for (unsigned int row_index = 0; row_index < this -> lidar -> get_row_count(); ++row_index) {

		for (unsigned int col_index = 1; col_index < this -> lidar  -> get_col_count(); ++col_index) {
			Ray * ray = this -> lidar -> get_ray(row_index, col_index);

			// If either the true target or the a-priori
			// shape were missed, this measurement is
			// unusable
			if (std::abs(ray -> get_range_residual()) > 1e10) {
				continue;
			}

			// Else, the measurement is used
			unsigned int v0_index = std::distance(
			                            this -> estimated_shape_model -> get_vertices() -> begin(),
			                            std::find(
			                                this -> estimated_shape_model -> get_vertices() -> begin(),
			                                this -> estimated_shape_model -> get_vertices() -> end(),
			                                ray -> get_computed_hit_facet() -> get_vertices() -> at(0))
			                        );



			unsigned int v1_index = std::distance(
			                            this -> estimated_shape_model -> get_vertices() -> begin(),
			                            std::find(this -> estimated_shape_model -> get_vertices() -> begin(),
			                                      this -> estimated_shape_model -> get_vertices() -> end(),
			                                      ray -> get_computed_hit_facet() -> get_vertices() -> at(1))


			                        );



			unsigned int v2_index = std::distance(
			                            this -> estimated_shape_model -> get_vertices() -> begin(),
			                            std::find(this -> estimated_shape_model -> get_vertices() -> begin(),
			                                      this -> estimated_shape_model -> get_vertices() -> end(),
			                                      ray -> get_computed_hit_facet() -> get_vertices() -> at(2))
			                        );

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
			std::vector<arma::rowvec> partials = this -> partial_range_partial_coordinates(P, u, ray -> get_computed_hit_facet());

			// The corresponding partitions of H_tilde are set
			H_tilde.cols(v0_index, v0_index + 2) = partials[0];
			H_tilde.cols(v1_index, v1_index + 2) = partials[1];
			H_tilde.cols(v2_index, v2_index + 2) = partials[2];

			// The Information matrix and the normal matrices are augmented
			info_mat += H_tilde.t() * H_tilde;
			normal_mat += H_tilde.t() * ray -> get_range_residual();


			// Htilde is reset!
			H_tilde = 0 * H_tilde;

		}
	}

	// The deviation in the vertices is computed
	arma::vec dV = arma::solve(info_mat, normal_mat);
	std::cout << arma::norm(dV) << std::endl;

}


std::vector<arma::rowvec> Filter::partial_range_partial_coordinates(const arma::vec & P, const arma::vec & u, Facet * facet) {

	std::vector<arma::rowvec> partials;

	arma::vec * n = facet -> get_facet_normal();

	std::vector<std::shared_ptr<Vertex > > * vertices = facet -> get_vertices();

	arma::vec * V0 =  vertices -> at(0) -> get_coordinates();
	arma::vec * V1 =  vertices -> at(1) -> get_coordinates();
	arma::vec * V2 =  vertices -> at(2) -> get_coordinates();

	arma::rowvec drhodV0 = ((*n).t()) / arma::dot(u, *n) + (*V0 - P).t() / arma::dot(u, *n) * (arma::eye<arma::mat>(3, 3)
	                       - (*n) * u.t() / arma::dot(u, *n)) * tilde(*V2 - *V1);


	arma::rowvec drhodV1 = (*V0 - P).t() / arma::dot(u, *n) * (arma::eye<arma::mat>(3, 3)
	                       - (*n) * u.t() / arma::dot(u, *n)) * tilde(*V0 - *V2);


	arma::rowvec drhodV2 = (*V0 - P).t() / arma::dot(u, *n) * (arma::eye<arma::mat>(3, 3)
	                       - (*n) * u.t() / arma::dot(u, *n)) * tilde(*V1 - *V0);

	partials.push_back(drhodV0);
	partials.push_back(drhodV1);
	partials.push_back(drhodV2);

	return partials;





}

