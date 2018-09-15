#include "IterativeClosestPoint.hpp"



IterativeClosestPoint::IterativeClosestPoint(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source) : ICPBase(pc_destination,pc_source){

}

double IterativeClosestPoint::compute_rms_residuals(
	const std::vector<PointPair> & point_pairs,
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D ) const {

	double J = 0;
	double mean = IterativeClosestPoint::compute_mean_residuals(point_pairs,dcm_S ,x_S ,dcm_D , x_D );
	#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
	for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {
		J += std::pow(IterativeClosestPoint::compute_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D) - mean,2);
	}
	return std::sqrt(J / (point_pairs.size()-1) );

}

double IterativeClosestPoint::compute_mean_residuals(
	const std::vector<PointPair> & point_pairs,
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D ) const {

	double J = 0;

	#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
	for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {

		J += IterativeClosestPoint::compute_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D)/ point_pairs.size();

	}

	return J;

}

double IterativeClosestPoint::compute_distance(const PointPair & point_pair, 
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D ) const{

	return arma::norm(dcm_S * point_pair.first -> get_point() + x_S 
		- dcm_D * point_pair.second -> get_point() - x_D) ;

}

double IterativeClosestPoint::compute_rms_residuals(
	const arma::mat::fixed<3,3> & dcm,
	const arma::vec::fixed<3> & x) {

	return IterativeClosestPoint::compute_rms_residuals(this -> point_pairs,dcm,x);

}




void IterativeClosestPoint::compute_pairs(int h,const arma::mat::fixed<3,3> & dcm ,const arma::vec::fixed<3> & x ) {


	if (use_true_pairs){

		if (this -> pc_source -> get_size() != this -> pc_destination -> get_size()){
			throw(std::runtime_error("Can't pair point clouds one-to-one since they are of different size"));
		}

		this -> point_pairs.clear();
		double p = std::log2(this -> pc_source -> get_size());
		int N_pairs_max = (int)(std::pow(2, std::max(p - h,0.)));

		arma::uvec random_source_indices = arma::linspace<arma::uvec>(0, this -> pc_source -> get_size() - 1,this -> pc_source -> get_size());
		random_source_indices = arma::shuffle(random_source_indices);

		for (int i = 0; i < N_pairs_max; ++i){
			this -> point_pairs.push_back(std::make_pair(this -> pc_source -> get_point(random_source_indices(i)),
				this -> pc_destination -> get_point(random_source_indices(i))));

		}

	}
	else{
		IterativeClosestPoint::compute_pairs(this -> point_pairs,this -> pc_source,this -> pc_destination,h, dcm,x);
	}

}

void IterativeClosestPoint::compute_pairs(
	std::vector<PointPair> & point_pairs,
	std::shared_ptr<PC> source_pc,
	std::shared_ptr<PC> destination_pc, 
	int h,
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D ){

	point_pairs.clear();

	std::map<double, PointPair > all_pairs;

	// int N_points = (int)(source_pc -> get_size() / std::pow(2, h));
	double p = std::log2(source_pc -> get_size());
	int N_pairs_max = (int)(std::pow(2, std::max(p - h,0.)));

	#if ICP_DEBUG
	std::cout << "\tMaking pairs at h = " << h << "\n";
	std::cout << "\tUsing a-priori transform:" << std::endl;
	std::cout << "\t\t X_S: " << x_S.t();
	std::cout << "\t\t MRP_S: " << RBK::dcm_to_mrp(dcm_S).t() ;
	std::cout << "\t\t X_D: " << x_D.t();
	std::cout << "\t\t MRP_D: " << RBK::dcm_to_mrp(dcm_D).t();
	#endif

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
		destination_source_dist_vector[i].first = closest_destination_point;
	}

	// The destination point is mapped to the source frame using the inverse of the a-priori transform
	// Then, the source pc is queried for the closest source point
	// to the mapped destination point
	// This double mapping process gets rid of spurious edge points

	#pragma omp parallel for
	for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {
		if (destination_source_dist_vector[i].first != nullptr){
			arma::vec test_destination_point = dcm_S.t() * ( dcm_D * destination_source_dist_vector[i].first -> get_point() + x_D - x_S);
			std::shared_ptr<PointNormal> closest_source_point = source_pc -> get_closest_point(test_destination_point);
			destination_source_dist_vector[i].second = closest_source_point;
		}
	}


	// The source/destination pairs are pre-formed
	std::vector<std::pair<unsigned int , double> > formed_pairs;
	for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {
		if (destination_source_dist_vector[i].first != nullptr){
			arma::vec S = dcm_S * destination_source_dist_vector[i].second -> get_point() + x_S;
			arma::vec D = dcm_D * destination_source_dist_vector[i].first -> get_point() + x_D;
			formed_pairs.push_back(std::make_pair(i,std::pow(arma::norm(S - D),2)));
		}
	}	


	#if ICP_DEBUG
	std::cout << "\tFormed " << formed_pairs.size() << " pairs before pruning\n";
	#endif

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

		if (std::abs(dist_vec(i) - mean) <= sd){
			point_pairs.push_back(
				std::make_pair(destination_source_dist_vector[formed_pairs[i].first].second,
					destination_source_dist_vector[formed_pairs[i].first].first));
		}

	}

	#if ICP_DEBUG
	std::cout << "\tMean pair distance : " << mean << std::endl;
	std::cout << "\tDistance sd : " << sd << std::endl;
	std::cout << "\tKept " << point_pairs.size() << " pairs\n";
	#endif

}


void IterativeClosestPoint::build_matrices(const int pair_index,const arma::vec::fixed<3> & mrp, 
		const arma::vec::fixed<3> & x,arma::mat::fixed<6,6> & info_mat_temp,
		arma::vec::fixed<6> & normal_mat_temp){


	arma::vec::fixed<3> S_i = this -> point_pairs[pair_index].first -> get_point();
	arma::vec::fixed<3> D_i = this -> point_pairs[pair_index].second -> get_point();

	// The partial derivative of the observation model is computed
	arma::mat::fixed<3,6> H = arma::zeros<arma::mat>(3,6);

	H.submat(0,0,2,2) = - arma::eye<arma::mat>(3,3);
	H.submat(0,3,2,5) = - 4 * RBK::tilde(RBK::mrp_to_dcm(mrp) * S_i);

	info_mat_temp = H.t() * H;

	normal_mat_temp = H.t() * (RBK::mrp_to_dcm(mrp) * S_i + x - D_i);
	
}
