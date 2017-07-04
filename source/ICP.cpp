#include "ICP.hpp"

ICP::ICP(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source) {

	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;


	this -> register_pc_mrp_multiplicative_partials(30,
	        1e-3,
	        1e-3,
	        true );


}


std::vector<std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > * ICP::get_point_pairs() {

	return &this -> point_pairs;

}


double ICP::compute_rms_residuals(
    const arma::mat & dcm,
    const arma::vec & x) {

	double J = 0;
	for (unsigned int pair_index = 0; pair_index != this -> point_pairs.size(); ++pair_index) {


		arma::vec source_point = *this -> point_pairs[pair_index].first -> get_point();
		arma::vec destination_point = *this -> point_pairs[pair_index].second -> get_point();
		arma::vec normal = *this -> point_pairs[pair_index].second -> get_normal();


		J += pow(arma::dot(dcm * source_point
		                   + x - destination_point,
		                   normal), 2); ;

	}

	J = std::sqrt(J / this -> point_pairs.size() );
	return J;

}



arma::vec ICP::get_X() const {
	return this -> X;
}

arma::mat ICP::get_DCM() const {
	return this -> DCM;
}



void ICP::register_pc_mrp_multiplicative_partials(
    const unsigned int iterations_max,
    const double rel_tol,
    const double stol,
    const bool pedantic) {

	double J;
	double J_0;
	double J_previous = std::numeric_limits<double>::infinity();

	// The batch estimator is initialized with a zero translation/zero rotation
	arma::vec mrp = {0, 0, 0};
	arma::vec x = {0, 0, 0};

	int h = (int)(std::log(this -> pc_source -> get_size()) / std::log(2) / 2);

	bool exit = false;

	while (h >= 0 && exit == false) {
		std::cout << "Hierchical level : " << std::to_string(h) << std::endl;

		// The ICP is iterated
		for (unsigned int iter = 0; iter < iterations_max; ++iter) {

			// The pairs are formed
			this -> compute_pairs_closest_minimum_distance(
			    mrp_to_dcm(mrp),
			    x, h);

			if (iter == 0) {

				// The initial residuals are computed
				J_0 = this -> compute_rms_residuals(mrp_to_dcm(mrp),
				                                    x);
				J = J_0;

			}

			// The matrices of the LS problem are now accumulated
			arma::mat Info_mat = arma::mat(6, 6);
			arma::vec Normal_mat = arma::vec(6);

			for (unsigned int pair_index = 0; pair_index != this -> point_pairs.size(); ++pair_index) {

				arma::mat Info_mat_first_rows = arma::mat(3, 6);
				arma::mat Info_mat_second_rows = arma::mat(3, 6);
				arma::mat Info_mat_temp = arma::mat(6, 6);

				arma::vec P_i = *this -> point_pairs[pair_index].first -> get_point();
				arma::vec Q_i = *this -> point_pairs[pair_index].second -> get_point();
				arma::vec n_i = *this -> point_pairs[pair_index].second -> get_normal();


				arma::vec Normal_mat_temp = arma::vec(6);

				// The partial derivative of the observation model is computed
				arma::mat H = this -> dGdSigma_multiplicative(mrp, P_i, n_i);

				// The information matrix is constructed by concatenating two 3-by-6 matrices
				Info_mat_first_rows(arma::span(), arma::span(0, 2)) = H.t() * H;
				Info_mat_first_rows(arma::span(), arma::span(3, 5)) = H.t() * n_i.t();
				Info_mat_second_rows(arma::span(), arma::span(0, 2)) = n_i * H ;
				Info_mat_second_rows(arma::span(), arma::span(3, 5)) = n_i * n_i.t();

				Info_mat_temp(arma::span(0, 2), arma::span()) = Info_mat_first_rows;
				Info_mat_temp(arma::span(3, 5), arma::span()) = Info_mat_second_rows;

				// The prefit residuals are computed
				double y_i = arma::dot(n_i.t(), Q_i -  mrp_to_dcm(mrp) * P_i - x );

				// The normal matrix is similarly built
				Normal_mat_temp(arma::span(0, 2)) = H.t() * y_i;
				Normal_mat_temp(arma::span(3, 5)) = n_i * y_i;

				if (pair_index == 0) {
					Info_mat = Info_mat_temp;
					Normal_mat = Normal_mat_temp;
				}
				else {
					Info_mat += Info_mat_temp;
					Normal_mat += Normal_mat_temp;
				}
			}


			// The state deviation [dmrp,dx] is solved for
			arma::vec dX = arma::solve(Info_mat, Normal_mat);
			arma::vec dmrp = {dX(0), dX(1), dX(2)};
			arma::vec dx = {dX(3), dX(4), dX(5)};

			// The state is updated
			mrp = dcm_to_mrp(mrp_to_dcm(dmrp) * mrp_to_dcm(mrp));

			x = x + dx;

			// the mrp is switched to its shadow if need be
			if (arma::norm(mrp) > 1) {
				mrp = - mrp / ( pow(arma::norm(mrp), 2));
			}


			// The postfit residuals are computed
			J = this -> compute_rms_residuals(mrp_to_dcm(mrp),
			                                  x);

			if (pedantic == true) {
				std::cout << "Pairs : " << this -> point_pairs.size() << std::endl;
				std::cout << "Residuals: " << J << std::endl;
			}


			if ( J / J_0 < rel_tol ) {
				exit = true;
				break;
			}

			if ( std::abs(J - J_previous) / J_previous < stol ) {
				h = h - 1;
				break;
			}

			else if (iter == iterations_max - 1) {
				h = h - 1;
				break;
			}

			// The postfit residuals become the prefit residuals of the next iteration
			J_previous = J;

		}
	}

	this -> X = x;
	this -> DCM = mrp_to_dcm(mrp);

	// Closest-point pairs are formed
	this -> compute_pairs_closest_minimum_distance(
	    this -> DCM,
	    this -> X, 0);



}


arma::rowvec ICP::dGdSigma_multiplicative(const arma::vec & mrp, const arma::vec & P, const arma::vec & n) {

	arma::rowvec partial = - 4 * P.t() * mrp_to_dcm(mrp).t() * tilde(n);
	return partial;

}

arma::mat ICP::compute_inertia(std::shared_ptr<PC> pc) {

	arma::mat cov = arma::zeros<arma::mat>(3, 3);


	for (unsigned int index = 0; index < pc -> get_size(); ++index) {
		arma::vec coords = pc -> get_point_coordinates(index);
		cov += arma::dot(coords,
		                 coords) * arma::eye<arma::mat>(3, 3) - coords * coords.t();
	}

	return cov;

}

void ICP::compute_pairs_closest_minimum_distance(
    const arma::mat & dcm,
    const arma::mat & x,
    int h) {


	this -> point_pairs.clear();

	std::map<double, std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > all_pairs;
	double mean_dist = 0;

	int N_points = (int)(this -> pc_source -> get_size() / std::pow(2, h));
	arma::ivec random_indices = arma::unique(arma::randi<arma::ivec>(N_points, arma::distr_param(0, this -> pc_source -> get_size() - 1)));


	std::map < std::shared_ptr<PointNormal> , std::map<double, std::shared_ptr<PointNormal> > > destination_to_source_pre_pairs;


	for (unsigned int i = 0; i < random_indices.n_rows; ++i) {

		arma::vec test_source_point = dcm * this -> pc_source -> get_point_coordinates(random_indices(i)) + x;

		std::shared_ptr<PointNormal> closest_destination_point = this -> pc_destination -> get_closest_point(test_source_point);

		std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > pair(this -> pc_source -> get_point(random_indices(i)), closest_destination_point);

		double dist = arma::norm(test_source_point - *closest_destination_point -> get_point());

		if (destination_to_source_pre_pairs.find(closest_destination_point) == destination_to_source_pre_pairs.end()) {

			std::map<double, std::shared_ptr<PointNormal> > distance_source_map;
			distance_source_map[dist] = this -> pc_source -> get_point(random_indices(i));
			destination_to_source_pre_pairs[closest_destination_point] = distance_source_map;

		}

		else {
			destination_to_source_pre_pairs[closest_destination_point][dist] = this -> pc_source -> get_point(random_indices(i));
		}

	}

	// Each destination point can only be paired once
	for (auto it = destination_to_source_pre_pairs.begin() ; it != destination_to_source_pre_pairs.end(); ++it) {

		std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > pair(it -> second . begin() -> second, it -> first);
		double dist = it -> second . begin() -> first;
		all_pairs[dist] = pair;

		mean_dist += dist;
	}


	mean_dist = mean_dist / all_pairs.size();

	// Only pairs whose residuals is less than the mean are kept
	for (auto it = all_pairs.begin(); it != all_pairs.end(); ++it) {
			this -> point_pairs.push_back(it -> second);
	}

}

void ICP::compute_pairs_closest_compatible_minimum_point_to_plane_dist(
    const arma::mat & dcm,
    const arma::mat & x,
    int h) {



	this -> point_pairs.clear();

	std::map<double, std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > all_pairs;
	double mean_dist = 0;

	int N_points = (int)(this -> pc_source -> get_size() / std::pow(2, h));
	arma::ivec random_indices = arma::unique(arma::randi<arma::ivec>(N_points, arma::distr_param(0, this -> pc_source -> get_size() - 1)));

	std::cout << "Number of random indices : " << std::to_string(random_indices.n_rows) << std::endl;

	std::map < std::shared_ptr<PointNormal> , std::map<double, std::shared_ptr<PointNormal> > > destination_to_source_pre_pairs;


	for (unsigned int i = 0; i < random_indices.n_rows; ++i) {

		arma::vec test_source_point = dcm * this -> pc_source -> get_point_coordinates(random_indices(i)) + x;

		std::shared_ptr<PointNormal> closest_destination_point = this -> pc_destination -> get_closest_point(test_source_point);

		arma::vec n_dest = *closest_destination_point -> get_normal();
		arma::vec n_source_transformed = dcm * (this -> pc_source -> get_point_normal(random_indices(i)));

		// If the two normals are compatible, the points are matched
		if (arma::dot(n_dest,
		              n_source_transformed) > std::sqrt(2) / 2 ) {

			double dist = std::sqrt(std::pow(arma::dot(n_dest, dcm * test_source_point + x - *closest_destination_point -> get_point()), 2));

			if (destination_to_source_pre_pairs.find(closest_destination_point) == destination_to_source_pre_pairs.end()) {

				std::map<double, std::shared_ptr<PointNormal> > distance_source_map;
				distance_source_map[dist] = this -> pc_source -> get_point(random_indices(i));
				destination_to_source_pre_pairs[closest_destination_point] = distance_source_map;

			}

			else {
				destination_to_source_pre_pairs[closest_destination_point][dist] = this -> pc_source -> get_point(random_indices(i));
			}

		}
	}


	// Each destination point can only be paired once
	for (auto it = destination_to_source_pre_pairs.begin() ; it != destination_to_source_pre_pairs.end(); ++it) {

		std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > pair(it -> second . begin() -> second, it -> first);
		double dist = it -> second . begin() -> first;
		all_pairs[dist] = pair;

		mean_dist += dist;
	}


	
	for (auto it = all_pairs.begin(); it != all_pairs.end(); ++it) {
		this -> point_pairs.push_back(it -> second);
	}



}

