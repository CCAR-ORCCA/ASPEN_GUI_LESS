#include "ICP.hpp"


ICP::ICP(std::shared_ptr<PC> pc_destination, 
	std::shared_ptr<PC> pc_source,
	arma::mat dcm_0,
	arma::vec X_0,
	bool verbose) {

	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;
	
	auto start = std::chrono::system_clock::now();

	
	this -> register_pc_mrp_multiplicative_partials(100,1e-8,1e-8,dcm_0,X_0);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;

	if (verbose){
		std::cout << "- Time elapsed in ICP: " << elapsed_seconds.count()<< " s"<< std::endl;
	}


}


std::vector<PointPair > * ICP::get_point_pairs() {

	return &this -> point_pairs;

}



double ICP::compute_normal_distance(const PointPair & point_pair,  
	const arma::mat & dcm_S,
	const arma::vec & x_S,
	const arma::mat & dcm_D,
	const arma::vec & x_D ){

	return arma::dot(dcm_S * point_pair.first -> get_point() + x_S 
		- dcm_D * point_pair.second -> get_point() - x_D,
		dcm_D * point_pair.second -> get_normal());

}


double ICP::compute_rms_residuals(
	const std::vector<PointPair> & point_pairs,  
	const arma::mat & dcm_S,
	const arma::vec & x_S,
	const arma::mat & dcm_D,
	const arma::vec & x_D) {

	double J = 0;

	#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
	for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {

		J += std::pow(ICP::compute_normal_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D),2);

	}

	J = std::sqrt(J / point_pairs.size() );
	return J;

}


double ICP::compute_mean_residuals(
	const std::vector<PointPair> & point_pairs,
	const arma::mat & dcm_S,
	const arma::vec & x_S,
	const arma::mat & dcm_D,
	const arma::vec & x_D) {

	double J = 0;

	#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
	for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {

		J += ICP::compute_normal_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D);

	}

	J = J / point_pairs.size();
	return J;

}


double ICP::compute_rms_residuals(
	const arma::mat & dcm,
	const arma::vec & x) {

	return ICP::compute_rms_residuals(this -> point_pairs,dcm,x);

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
	arma::mat dcm_0,
	arma::vec X_0,
	bool verbose) {

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

		// The ICP is iterated
		for (unsigned int iter = 0; iter < iterations_max; ++iter) {

			if ( next_h == true ) {
				// The pairs are formed only after a change in the hierchical search
				

				#if USE_OMP_ICP
				
				this -> compute_pairs(h,RBK::mrp_to_dcm(mrp),x);
				
				#else
				throw(std::runtime_error("Should deprecate compute_pairs_closest_compatible_minimum_point_to_plane_dist"));
				this -> compute_pairs_closest_compatible_minimum_point_to_plane_dist(
					RBK::mrp_to_dcm(mrp),
					x, h);
				
				#endif


				
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


			#if ICP_DEBUG
			std::cout << "Hierchical level : " << std::to_string(h) << std::endl;
			std::cout << "Number of valid pairs: " <<  this -> point_pairs.size() << std::endl;
			#endif


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
				H = ICP::dGdSigma_multiplicative(mrp, P_i, n_i);

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

			#if ICP_DEBUG
			std::cout << "\nInfo mat: " << std::endl;
			std::cout << Info_mat << std::endl;
			std::cout << "\nNormal mat: " << std::endl;
			std::cout << Normal_mat << std::endl;
			std::cout << "\nDeviation : " << std::endl;
			std::cout << dX << std::endl;
			std::cout << "\nResiduals: " << J << std::endl;
			std::cout << "MRP: \n" << mrp << std::endl;
			std::cout << "x: \n" << x << std::endl;
			std::cout << "Covariance :\n" << std::endl;
			std::cout << arma::inv(Info_mat) << std::endl;
			#endif


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


void ICP::compute_pairs(int h,const arma::mat & dcm,const arma::vec & x) {

	ICP::compute_pairs(this -> point_pairs,this -> pc_source,this -> pc_destination,h, dcm,x);
}

void ICP::compute_pairs(
	std::vector<PointPair> & point_pairs,
	std::shared_ptr<PC> source_pc,
	std::shared_ptr<PC> destination_pc, 
	int h,
	const arma::mat & dcm_S,
	const arma::vec & x_S,
	const arma::mat & dcm_D,
	const arma::vec & x_D){

	point_pairs.clear();

	std::map<double, PointPair > all_pairs;

	// int N_points = (int)(source_pc -> get_size() / std::pow(2, h));
	double p = std::log2(source_pc -> get_size());
	int N_points = (int)(std::pow(2, p - h));

	// a maximum of $N_points pairs will be formed. $N_points points are extracted from the source point cloud
	arma::ivec random_source_indices = arma::unique(arma::randi<arma::ivec>(N_points, arma::distr_param(0, source_pc -> get_size() - 1)));
	std::vector<PointPair> destination_source_dist_vector;

	for (unsigned int i = 0; i < random_source_indices.n_rows; ++i) {
		PointPair destination_source_dist_pair = std::make_pair(nullptr,source_pc -> get_point(random_source_indices(i)));
		destination_source_dist_vector.push_back(destination_source_dist_pair);
	}

	// The $N_points half-pair we defined are mapped to the destination frame using the 
	// a-priori transform. Then, the destination pc is queried for the closest destination point
	// to the mapped source point
	#pragma omp parallel for
	for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {

		arma::vec test_source_point = dcm_D.t() * (dcm_S * destination_source_dist_vector[i].second -> get_point() + x_S - x_D);

		std::shared_ptr<PointNormal> closest_destination_point = destination_pc -> get_closest_point(test_source_point);

		arma::vec n_dest = dcm_D * closest_destination_point -> get_normal();
		arma::vec n_source = dcm_S * destination_source_dist_vector[i].second -> get_normal();

		// If the two normals are compatible, the points are matched
		if (arma::dot(n_dest,n_source) > std::sqrt(2) / 2 ) {
			destination_source_dist_vector[i].first = closest_destination_point;

		}
	}


	// The destination point is mapped to the source frame using the inverse of the a-priori transform
	// Then, the source pc is queried for the closest source point
	// to the mapped destination point
	// This double mapping process gets rid of edge points
	#pragma omp parallel for
	for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {
		if (destination_source_dist_vector[i].first != nullptr){
			arma::vec test_destination_point = dcm_S.t() * ( dcm_D * destination_source_dist_vector[i].first -> get_point() + x_D - x_S);

			std::shared_ptr<PointNormal> closest_source_point = source_pc -> get_closest_point(test_destination_point);

			arma::vec n_source = dcm_S * closest_source_point -> get_normal();
			arma::vec n_destination = dcm_D * destination_source_dist_vector[i].first -> get_normal();

		// If the two normals are compatible, the points are matched
			if (arma::dot(n_source,n_destination) > std::sqrt(2) / 2 ) {
				destination_source_dist_vector[i].second = closest_source_point;
			}
		}

	}


	// The source/destination pairs are pre-formed
	std::vector<std::pair<unsigned int , double> > formed_pairs;

	for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {
		
		if (destination_source_dist_vector[i].first != nullptr){
			arma::vec S = dcm_S * destination_source_dist_vector[i].second -> get_point() + x_S;
			arma::vec n = dcm_D * destination_source_dist_vector[i].first -> get_normal();
			arma::vec D = dcm_D * destination_source_dist_vector[i].first -> get_point() + x_D;

			formed_pairs.push_back(std::make_pair(i,std::pow(arma::dot(n,S - D),2)));
		}
	}	

	// Pairing error statistics are collected
	arma::vec dist_vec(formed_pairs.size());
	if (formed_pairs.size()== 0){
		throw(ICPNoPairsException());
	}
	#pragma omp parallel for
	for (unsigned int i = 0; i < dist_vec.n_rows; ++i) {
		dist_vec(i) = formed_pairs[i].second;
	}

	// Only pairs featuring an error that is less a one sd threshold are inliers 
	// included in the final pairing
	double mean = arma::mean(dist_vec);
	double sd = arma::stddev(dist_vec);
	
	for (unsigned int i = 0; i < dist_vec.n_rows; ++i) {

		if (std::abs(dist_vec(i) - mean) < sd){
			point_pairs.push_back(
				std::make_pair(destination_source_dist_vector[formed_pairs[i].first].second,
					destination_source_dist_vector[formed_pairs[i].first].first));
		}

	}



}

