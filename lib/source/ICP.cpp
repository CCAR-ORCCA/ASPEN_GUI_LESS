#include "ICP.hpp"
#include "CustomReductions.hpp"

ICP::ICP(std::shared_ptr<PC> pc_destination, 
	std::shared_ptr<PC> pc_source,
	bool verbose,
	const arma::mat & M_save,
	const arma::vec & X_save) {

	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;

	this -> M_save = M_save;
	this -> X_save = X_save;

	

}

void ICP::set_use_true_pairs(bool use_true_pairs){
	this -> use_true_pairs = use_true_pairs;
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
	return std::sqrt(J / point_pairs.size() );

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

		J += ICP::compute_normal_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D)/ point_pairs.size();

	}

	return J;

}


double ICP::compute_rms_residuals(
	const arma::mat & dcm,
	const arma::vec & x) {

	return ICP::compute_rms_residuals(this -> point_pairs,dcm,x);

}



arma::vec ICP::get_x() const {
	return this -> X;
}

arma::mat ICP::get_dcm() const {
	return this -> DCM;
}

void ICP::set_iterations_max(unsigned int iterations_max){
	this -> iterations_max = iterations_max;
}

void ICP::register_pc(
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

	arma::mat::fixed<6,6> info_mat;
	arma::vec::fixed<6> normal_mat;

	while (h >= 0 && exit == false) {

		// The ICP is iterated
		for (unsigned int iter = 0; iter < this -> iterations_max; ++iter) {

			if ( next_h == true ) {
				// The pairs are formed only after a change in the hierchical search

				this -> compute_pairs(h,RBK::mrp_to_dcm(mrp),x);
				
				next_h = false;
			}

			if (iter == 0 ) {

				// The initial residuals are computed
				J_0 = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),
					x);
				J = J_0;

			}


			// The matrices of the LS problem are now accumulated
			info_mat.fill(0);
			normal_mat.fill(0);


			#if ICP_DEBUG
			std::cout << "Hierchical level : " << std::to_string(h) << std::endl;
			std::cout << "Number of valid pairs: " <<  this -> point_pairs.size() << std::endl;
			#endif


			#pragma omp parallel for reduction(+:info_mat), reduction(+:normal_mat) if (USE_OMP_ICP)
			for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {
				
				arma::mat::fixed<6,6> info_mat_temp;
				arma::vec::fixed<6> normal_mat_temp;
				arma::vec::fixed<3> P_i,Q_i,n_i;
				arma::rowvec::fixed<3> H;

				P_i = this -> point_pairs[pair_index].first -> get_point();
				Q_i = this -> point_pairs[pair_index].second -> get_point();
				n_i = this -> point_pairs[pair_index].second -> get_normal();

				// The partial derivative of the observation model is computed
				H = ICP::dGdSigma_multiplicative(mrp, P_i, n_i);

				info_mat_temp.submat(0,0,2,2) = H.t() * H;
				info_mat_temp.submat(0,3,2,5) = H.t() * n_i.t();
				info_mat_temp.submat(3,0,5,2) = n_i * H ;
				info_mat_temp.submat(3,3,5,5) = n_i * n_i.t();

				// The prefit residuals are computed
				double y_i = arma::dot(n_i.t(), Q_i -  RBK::mrp_to_dcm(mrp) * P_i - x );

				// The normal matrix is similarly built
				normal_mat_temp.subvec(0, 2) = H.t() * y_i;
				normal_mat_temp.subvec(3, 5) = n_i * y_i;

				info_mat += info_mat_temp;
				normal_mat += normal_mat_temp;
				
			}


			#if ICP_DEBUG
			std::cout << "\nInfo mat: " << std::endl;
			std::cout << info_mat << std::endl;
			std::cout << "\nNormal mat: " << std::endl;
			std::cout << normal_mat << std::endl;
			#endif

			// The state deviation [dmrp,dx] is solved for
			arma::vec dX = arma::solve(info_mat, normal_mat);
			arma::vec dmrp = dX.subvec(0,2);
			arma::vec dx = dX.subvec(3,5);

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
			std::cout << "\nDeviation : " << std::endl;
			std::cout << dX << std::endl;
			std::cout << "\nResiduals: " << J << std::endl;
			std::cout << "MRP: \n" << mrp << std::endl;
			std::cout << "x: \n" << x << std::endl;
			std::cout << "Covariance :\n" << std::endl;
			std::cout << arma::inv(info_mat) << std::endl;
			#endif


			if ( J / J_0 <= rel_tol || J == 0) {
				exit = true;

				break;
			}

			if ( std::abs(J - J_previous) / J <= stol ) {
				h = h - 1;
				next_h = true;

				J_previous = std::numeric_limits<double>::infinity();

				break;
			}

			else if (iter + 1== this -> iterations_max ) {


				this -> pc_source -> save("../output/pc/crash.obj",this -> M_save * RBK::mrp_to_dcm(mrp),this -> M_save * x + this -> X_save);


				throw ICPException();

				break;
			}

			// The postfit residuals become the prefit residuals of the next iteration
			J_previous = J;

		}
	}

	this -> X = x;
	this -> DCM = RBK::mrp_to_dcm(mrp);
	this -> R = arma::inv(info_mat);
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

	if (use_true_pairs){

		if (this -> pc_source -> get_size() != this -> pc_destination -> get_size()){
			throw(std::runtime_error("Can't pair point clouds one-to-one since they are of different size"));
		}

		this -> point_pairs.clear();
		double p = std::log2(this -> pc_source -> get_size());
		int N_pairs_max = (int)(std::pow(2, p - h));

		arma::uvec random_source_indices = arma::linspace<arma::uvec>(0, this -> pc_source -> get_size() - 1,this -> pc_source -> get_size());
		random_source_indices = arma::shuffle(random_source_indices);

		for (int i = 0; i < N_pairs_max; ++i){
			this -> point_pairs.push_back(std::make_pair(this -> pc_source -> get_point(random_source_indices(i)),
				this -> pc_destination -> get_point(random_source_indices(i))));

		}

	}
	else{

		ICP::compute_pairs(this -> point_pairs,this -> pc_source,this -> pc_destination,h, dcm,x);
	}


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
	int N_pairs_max = (int)(std::pow(2, p - h));

	// a maximum of $N_pairs_max pairs will be formed. $N_points points are extracted from the source point cloud	
	arma::uvec random_source_indices = arma::linspace<arma::uvec>(0, source_pc -> get_size() - 1,source_pc -> get_size());
	random_source_indices = arma::shuffle(random_source_indices);

	std::vector<PointPair> destination_source_dist_vector;

	for (int i = 0; i < N_pairs_max; ++i) {
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

	#if ICP_DEBUG
	std::cout << "Mean pair distance : " << mean << std::endl;
	std::cout << "Distance sd : " << sd << std::endl;
	#endif

	for (unsigned int i = 0; i < dist_vec.n_rows; ++i) {

		if (std::abs(dist_vec(i) - mean) <= sd){
			point_pairs.push_back(
				std::make_pair(destination_source_dist_vector[formed_pairs[i].first].second,
					destination_source_dist_vector[formed_pairs[i].first].first));
		}

	}
}




