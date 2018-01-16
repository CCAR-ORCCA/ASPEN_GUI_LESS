#include "ICP.hpp"

ICP::ICP(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source,
	arma::mat dcm_0,
	arma::vec X_0,
	bool pedantic) {

	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;
	
	// auto start = std::chrono::system_clock::now();

	
	this -> register_pc_mrp_multiplicative_partials(100,
		1e-8,
		1e-8,
		pedantic, 
		dcm_0,
		X_0 );
	// auto end = std::chrono::system_clock::now();

	// std::chrono::duration<double> elapsed_seconds = end-start;

	// std::cout << "- Time elapsed in ICP: " << elapsed_seconds.count()<< " s"<< std::endl;
	


}


std::vector<std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > * ICP::get_point_pairs() {

	return &this -> point_pairs;

}


double ICP::compute_rms_residuals(
	const arma::mat & dcm,
	const arma::vec & x) {

	double J = 0;

	#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
	for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {

		arma::vec source_point = this -> point_pairs[pair_index].first -> get_point();
		arma::vec destination_point = this -> point_pairs[pair_index].second -> get_point();
		arma::vec normal = this -> point_pairs[pair_index].second -> get_normal();

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

arma::mat ICP::get_M() const {
	return this -> DCM;
}



void ICP::register_pc_mrp_multiplicative_partials(
	const unsigned int iterations_max,
	const double rel_tol,
	const double stol,
	const bool pedantic,
	arma::mat dcm_0,
	arma::vec X_0) {

	double J  = std::numeric_limits<double>::infinity();
	double J_0  = std::numeric_limits<double>::infinity();
	double J_previous = std::numeric_limits<double>::infinity();

	// The batch estimator is initialized
	arma::vec mrp = RBK::dcm_to_mrp(dcm_0);
	arma::vec x = X_0;

	int h = 7;

	bool exit = false;
	bool next_h = true;


	arma::mat::fixed<6,6> Info_mat;
	arma::vec::fixed<6> Normal_mat;


	while (h >= 0 && exit == false) {

		if (pedantic) {
			std::cout << "Hierchical level : " << std::to_string(h) << std::endl;
		}



		// The ICP is iterated
		for (unsigned int iter = 0; iter < iterations_max; ++iter) {

			if ( next_h == true ) {
				// The pairs are formed only after a change in the hierchical search
				this -> compute_pairs_closest_compatible_minimum_point_to_plane_dist(
					RBK::mrp_to_dcm(mrp),
					x, h);
				
				next_h = false;
			}

			if (iter == 0 ) {


				// The initial residuals are computed
				J_0 = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),
					x);
				J = J_0;

			}


			// The matrices of the LS problem are now accumulated
			Info_mat.fill(0);
			Normal_mat.fill(0);


			#pragma omp parallel for reduction(+:Normal_mat,Info_mat) if (USE_OMP_ICP)

			for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {

				arma::mat::fixed<6,6> Info_mat_temp;
				arma::vec::fixed<6> Normal_mat_temp;
				arma::vec::fixed<3> P_i,Q_i,n_i;
				arma::rowvec::fixed<3> H;

				P_i = this -> point_pairs[pair_index].first -> get_point();
				Q_i = this -> point_pairs[pair_index].second -> get_point();
				n_i = this -> point_pairs[pair_index].second -> get_normal();

				// The partial derivative of the observation model is computed
				H = this -> dGdSigma_multiplicative(mrp, P_i, n_i);

				Info_mat_temp(arma::span(0,2),arma::span(0,2)) = H.t() * H;
				Info_mat_temp(arma::span(0,2),arma::span(3,5)) = H.t() * n_i.t();
				Info_mat_temp(arma::span(3,5),arma::span(0,2)) = n_i * H ;
				Info_mat_temp(arma::span(3,5),arma::span(3,5)) = n_i * n_i.t();


				// The prefit residuals are computed
				double y_i = arma::dot(n_i.t(), Q_i -  RBK::mrp_to_dcm(mrp) * P_i - x );

				// The normal matrix is similarly built
				Normal_mat_temp.rows(0, 2) = H.t() * y_i;
				Normal_mat_temp.rows(3, 5) = n_i * y_i;

				
				Info_mat += Info_mat_temp;
				Normal_mat += Normal_mat_temp;
				
			}


			// The state deviation [dmrp,dx] is solved for
			arma::vec dX = arma::solve(Info_mat, Normal_mat);
			arma::vec dmrp = {dX(0), dX(1), dX(2)};
			arma::vec dx = {dX(3), dX(4), dX(5)};

			// The state is updated
			mrp = RBK::dcm_to_mrp(RBK::mrp_to_dcm(dmrp) * RBK::mrp_to_dcm(mrp));

			x = x + dx;

			// the mrp is switched to its shadow if need be
			if (arma::norm(mrp) > 1) {
				mrp = - mrp / ( pow(arma::norm(mrp), 2));
			}


			// The postfit residuals are computed
			J = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),
				x);

			if (pedantic == true) {
				std::cout << "Pairs : " << this -> point_pairs.size() << std::endl;
				std::cout << "Residuals: " << J << std::endl;
				std::cout << "MRP: " << mrp.t() << std::endl;
				std::cout << "x: " << x.t() << std::endl;
			}


			if ( J / J_0 < rel_tol ) {
				exit = true;

				break;
			}

			if ( std::abs(J - J_previous) / J < stol ) {
				h = h - 1;
				next_h = true;

				J_previous = std::numeric_limits<double>::infinity();

				break;
			}

			else if (iter == iterations_max - 1) {

				throw ICPException();

				break;
			}

			// The postfit residuals become the prefit residuals of the next iteration
			J_previous = J;

		}
	}

	this -> X = x;
	this -> DCM = RBK::mrp_to_dcm(mrp);
	this -> R = arma::inv(Info_mat);
	this -> J_res = J ;

}

arma::mat ICP::get_R() const {
	return this -> R;
}

double ICP::get_J_res() const {
	return this -> J_res;
}

arma::rowvec ICP::dGdSigma_multiplicative(const arma::vec & mrp, const arma::vec & P, const arma::vec & n) {

	arma::rowvec partial = - 4 * P.t() * RBK::mrp_to_dcm(mrp).t() * RBK::tilde(n);
	return partial;

}


void ICP::compute_pairs_closest_minimum_distance(
	const arma::mat & dcm,
	const arma::mat & x,
	int h) {

	this -> point_pairs.clear();

	std::map<double, std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > all_pairs;

	int N_points = (int)(this -> pc_source -> get_size() / std::pow(2, h));
	arma::ivec random_indices = arma::unique(arma::randi<arma::ivec>(N_points, arma::distr_param(0, this -> pc_source -> get_size() - 1)));


	std::map < std::shared_ptr<PointNormal> , std::map<double, std::shared_ptr<PointNormal> > > destination_to_source_pre_pairs;


	for (unsigned int i = 0; i < random_indices.n_rows; ++i) {

		arma::vec test_source_point = dcm * this -> pc_source -> get_point_coordinates(random_indices(i)) + x;

		std::shared_ptr<PointNormal> closest_destination_point = this -> pc_destination -> get_closest_point(test_source_point);

		std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > pair(this -> pc_source -> get_point(random_indices(i)), closest_destination_point);

		double dist = arma::norm(test_source_point - closest_destination_point -> get_point());

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
	}


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

	int N_points = (int)(this -> pc_source -> get_size() / std::pow(2, h));

	arma::ivec random_indices = arma::unique(arma::randi<arma::ivec>(N_points, arma::distr_param(0, this -> pc_source -> get_size() - 1)));

	std::map < std::shared_ptr<PointNormal> , std::map<double, std::shared_ptr<PointNormal> > > destination_to_source_pre_pairs;






	for (unsigned int i = 0; i < random_indices.n_rows; ++i) {

		arma::vec test_source_point = dcm * this -> pc_source -> get_point_coordinates(random_indices(i)) + x;

		std::shared_ptr<PointNormal> closest_destination_point = this -> pc_destination -> get_closest_point(test_source_point);

		arma::vec n_dest = closest_destination_point -> get_normal();
		arma::vec n_source_transformed = dcm * (this -> pc_source -> get_point_normal(random_indices(i)));

		// If the two normals are compatible, the points are matched
		if (arma::dot(n_dest,n_source_transformed) > std::sqrt(2) / 2 ) {

			double dist = std::sqrt(std::pow(arma::dot(n_dest, test_source_point - closest_destination_point -> get_point()), 2));

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
	for (auto it = destination_to_source_pre_pairs.begin() ;it != destination_to_source_pre_pairs.end(); ++it) {

		// Source/Destination pair
		std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > pair(it -> second . begin() -> second, it -> first);
		double dist = it -> second . begin() -> first;
		all_pairs[dist] = pair;

	}





	arma::vec dist_vec(all_pairs.size());

	for (unsigned int i = 0; i < dist_vec.n_rows; ++i) {
		dist_vec(i) = std::next(all_pairs.begin(), i) -> first;
	}

	double mean = arma::mean(dist_vec);

// Erase 
	double sd = arma::stddev(dist_vec);
	for (auto it = all_pairs.begin(); it != all_pairs.end(); ++it) {

		if (std::abs(it -> first - mean) < sd)
			this -> point_pairs.push_back(it -> second);



	}



}

// void ICP::compute_pairs_closest_compatible_minimum_point_to_plane_dist(
// 	const arma::mat & dcm,
// 	const arma::mat & x,
// 	int h) {


// 	this -> point_pairs.clear();

// 	std::map<double, std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > all_pairs;

// 	int N_points = (int)(this -> pc_source -> get_size() / std::pow(2, h));

// 	arma::ivec random_indices = arma::unique(arma::randi<arma::ivec>(N_points, arma::distr_param(0, this -> pc_source -> get_size() - 1)));

// 	std::map < std::shared_ptr<PointNormal> , std::map<double, std::shared_ptr<PointNormal> > > destination_to_source_pre_pairs;



// 	std::vector<std::pair<std::shared_ptr<PointNormal>,std::shared_ptr<PointNormal> > > source_to_destination_pre_pairs_vec;
// 	std::vector<double> dists;



// 	for (unsigned int i = 0; i < random_indices.n_rows; ++i) {

// 		source_to_destination_pre_pairs_vec.push_back(std::make_pair(this -> pc_source -> get_point(i),nullptr));
// 		dists.push_back(std::numeric_limits<double>::infinity());
// 	}



// 	#pragma omp parallel for 
// 	for (unsigned int i = 0; i < random_indices.n_rows; ++i) {


// 		arma::vec test_source_point = dcm * source_to_destination_pre_pairs_vec[i].first -> get_point() + x;

// 		std::shared_ptr<PointNormal> closest_destination_point = this -> pc_destination -> get_closest_point(test_source_point);

// 		arma::vec n_dest = closest_destination_point -> get_normal();
// 		arma::vec n_source_transformed = dcm * source_to_destination_pre_pairs_vec[i].first -> get_normal();

// 		// If the two normals are compatible, the points are matched
// 		if (arma::dot(n_dest,n_source_transformed) > std::sqrt(2) / 2 ) {

// 			double dist = std::sqrt(std::pow(arma::dot(n_dest, test_source_point - closest_destination_point -> get_point()), 2));

// 			source_to_destination_pre_pairs_vec[i].second = closest_destination_point;
// 			dists[i] = dist;
// 		}

// 	}

// 	// Each destination point can only be paired once
// 	for (unsigned int i = 0; i < random_indices.n_rows; ++i) {

// 		if (dists[i] < std::numeric_limits<double>::infinity()){
// 			// This is a valid source/destination pair

// 			all_pairs[dists[i]] = source_to_destination_pre_pairs_vec[i];
// 		}

// 	}

// 	arma::vec dist_vec(all_pairs.size());

// 	for (unsigned int i = 0; i < dist_vec.n_rows; ++i) {
// 		dist_vec(i) = std::next(all_pairs.begin(), i) -> first;
// 	}

// 	double mean = arma::mean(dist_vec);

// // Erase 
// 	double sd = arma::stddev(dist_vec);
// 	for (auto it = all_pairs.begin(); it != all_pairs.end(); ++it) {

// 		if (std::abs(it -> first - mean) < sd){
// 			this -> point_pairs.push_back(it -> second);
// 		}

// 	}


// }

