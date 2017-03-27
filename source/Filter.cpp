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

		// this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> propagate();
		// this -> frame_graph -> get_frame(this -> estimated_shape_model -> get_ref_frame_name()) -> propagate();

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

		this -> lidar -> send_flash(this -> true_shape_model,false);
		this -> lidar -> send_flash(this -> estimated_shape_model,true);

		this -> lidar -> plot_ranges("../images/true_" + std::to_string(time_index),0);
		this -> lidar -> plot_ranges("../images/computed_" + std::to_string(time_index),1);
		this -> lidar -> plot_ranges("../images/residuals_" + std::to_string(time_index),2);


		// this -> lidar -> plot_range_residuals("../images/residuals" + std::to_string(time_index));

	}


}
