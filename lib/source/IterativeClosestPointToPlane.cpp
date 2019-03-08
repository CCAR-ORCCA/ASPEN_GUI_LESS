#include "IterativeClosestPointToPlane.hpp"
#define ICP2P_DEBUG 1
#include <chrono>
#include <set>

IterativeClosestPointToPlane::IterativeClosestPointToPlane() : ICPBase(){

}

IterativeClosestPointToPlane::IterativeClosestPointToPlane(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source) : ICPBase(pc_destination,pc_source){

}

double IterativeClosestPointToPlane::compute_distance(const PointPair & point_pair, 
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D ) const{


	return IterativeClosestPointToPlane::compute_distance(point_pair, dcm_S ,x_S ,dcm_D ,x_D,
		this -> pc_source,this -> pc_destination) ;
}


double IterativeClosestPointToPlane::compute_distance(const PointPair & point_pair, 
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D,
	const std::shared_ptr<PC> & pc_S,
	const std::shared_ptr<PC> & pc_D){

	return arma::dot(dcm_S * pc_S -> get_point_coordinates(point_pair.first)  + x_S 
		- dcm_D * pc_D -> get_point_coordinates(point_pair.second)  - x_D,
		dcm_D * pc_D -> get_normal_coordinates(point_pair.second));
}

void IterativeClosestPointToPlane::compute_pairs(int h,
	const arma::mat::fixed<3,3> & dcm ,
	const arma::vec::fixed<3> & x ) {

	if (use_true_pairs){
		if (this -> pc_source -> size() != this -> pc_destination -> size()){
			throw(std::runtime_error("Can't pair point clouds one-to-one since they are of different size"));
		}

		this -> point_pairs.clear();
		double p = std::log2(this -> pc_source -> size());
		int N_pairs_max_from_source = (int)(std::pow(2, std::max(p - h,0.)));

		arma::ivec random_source_indices = arma::linspace<arma::ivec>(0, this -> pc_source -> size() - 1,this -> pc_source -> size());
		if (h != 0){
			random_source_indices = arma::shuffle(random_source_indices);
		}
		for (int i = 0; i < N_pairs_max_from_source; ++i){
			this -> point_pairs.push_back(std::make_pair<int,int>(random_source_indices(i),random_source_indices(i)));
			
		}

	}
	else{
		IterativeClosestPointToPlane::compute_pairs(this -> point_pairs,this -> pc_source,this -> pc_destination,h, dcm,x);
	}

}

void IterativeClosestPointToPlane::compute_pairs(
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

	// int N_points = (int)(source_pc -> size() / std::pow(2, h));
	int N_pairs_max_from_source = (int)(std::pow(2, std::max(std::log2(source_pc -> size()) - h,0.)));
	int N_pairs_max_from_destination = (int)(std::pow(2, std::max(std::log2(destination_pc -> size()) - h,0.)));


	#if ICP2P_DEBUG
	std::cout << "\tMaking " << std::min(N_pairs_max_from_source,N_pairs_max_from_destination) << " pairs at h = " << h << "\n";
	std::cout << "\tUsing a-priori transform:" << std::endl;
	std::cout << "\t\t X_S: " <<x_S.t();
	std::cout << "\t\t MRP_S: " << RBK::dcm_to_mrp(dcm_S).t() ;
	std::cout << "\t\t X_D: " << x_D.t();
	std::cout << "\t\t MRP_D: " << RBK::dcm_to_mrp(dcm_D).t();
	#endif	

	std::vector<std::pair<unsigned int , double> > formed_pairs;
	std::vector<PointPair> destination_source_dist_vector;


	if (N_pairs_max_from_source < N_pairs_max_from_destination){
		#if ICP2P_DEBUG
		std::cout<< "source_pc is the smallest point cloud\n";
		#endif

	// a maximum of $N_pairs_max_from_source pairs will be formed. $N_points points are extracted from the source point cloud	

		arma::ivec random_source_indices = arma::randi<arma::ivec>(N_pairs_max_from_source,arma::distr_param(0,source_pc -> size() - 1));


		for (int i = 0; i < N_pairs_max_from_source; ++i) {
			PointPair destination_source_dist_pair = std::make_pair(-1,random_source_indices(i));
			destination_source_dist_vector.push_back(destination_source_dist_pair);
		}

	#if ICP2P_DEBUG
		std::cout << "\t\tHaving a maximum of " << destination_source_dist_vector.size() << " pairs\n";
	#endif
	// The $N_points half-pair we defined are mapped to the destination frame using the 
	// a-priori transform. Then, the destination pc is queried for the closest destination point
	// to the mapped source point


	#pragma omp parallel for
		for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {

			arma::vec::fixed<3> test_source_point = dcm_D.t() * (dcm_S * source_pc -> get_point_coordinates(destination_source_dist_vector[i].second)+ x_S - x_D);
			int index_closest_destination_point = destination_pc -> get_closest_point(test_source_point);

			const PointNormal & closest_destination_point = destination_pc -> get_point(index_closest_destination_point);

			arma::vec::fixed<3> n_dest = dcm_D * closest_destination_point.get_normal_coordinates();
			arma::vec::fixed<3> n_source = dcm_S * source_pc -> get_normal_coordinates(destination_source_dist_vector[i].second);

		// If the two normals are compatible, the points are matched
			if (arma::dot(n_dest,n_source) > std::sqrt(2) / 2 ) {
				destination_source_dist_vector[i].first = index_closest_destination_point;

			}
		}



	// The destination point is mapped to the source frame using the inverse of the a-priori transform
	// Then, the source pc is queried for the closest source point
	// to the mapped destination point
	// This double mapping process gets rid of edge points



	#pragma omp parallel for
		for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {
			if (destination_source_dist_vector[i].first != -1){
				arma::vec test_destination_point = dcm_S.t() * ( dcm_D * destination_pc -> get_point_coordinates(destination_source_dist_vector[i].first) + x_D - x_S);


				int index_closest_source_point = source_pc -> get_closest_point(test_destination_point);
				arma::vec closest_source_point = source_pc -> get_point_coordinates(index_closest_source_point);

				arma::vec n_source = dcm_S * source_pc -> get_normal_coordinates(index_closest_source_point);
				arma::vec n_destination = dcm_D * destination_pc -> get_normal_coordinates(destination_source_dist_vector[i].first);

		// If the two normals are compatible, the points are matched
				if (arma::dot(n_source,n_destination) > std::sqrt(2) / 2 ) {
					destination_source_dist_vector[i].second = index_closest_source_point;
				}
			}

		}

	// The source/destination pairs are pre-formed

		for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {

			if (destination_source_dist_vector[i].first != -1){
				arma::vec S = dcm_S * source_pc -> get_point_coordinates(destination_source_dist_vector[i].second) + x_S;
				arma::vec n = dcm_D * destination_pc -> get_normal_coordinates(destination_source_dist_vector[i].first);
				arma::vec D = dcm_D * destination_pc -> get_point_coordinates(destination_source_dist_vector[i].first) + x_D;

				formed_pairs.push_back(std::make_pair(i,arma::dot(n,S - D)));
			}
		}	



	}
	else{



	#if ICP2P_DEBUG
		std::cout<< "destination_pc is the smallest point cloud\n";
		#endif
		// a maximum of $N_pairs_max_from_destination pairs will be formed. $N_points points are extracted from the destination point cloud	

		arma::ivec random_destination_indices = arma::randi<arma::ivec>(N_pairs_max_from_destination,arma::distr_param(0,destination_pc -> size() - 1));

		std::vector<PointPair> destination_source_dist_vector;

		for (int i = 0; i < N_pairs_max_from_destination; ++i) {
			PointPair destination_source_dist_pair = std::make_pair(random_destination_indices(i),-1);
			destination_source_dist_vector.push_back(destination_source_dist_pair);
		}


	#if ICP2P_DEBUG
		std::cout << "\t\tHaving a maximum of " << destination_source_dist_vector.size() << " pairs\n";
	#endif
	// The $N_points half-pair we defined are mapped to the destination frame using the 
	// a-priori transform. Then, the source pc is queried for the closest source point
	// to the mapped destination point


	#pragma omp parallel for
		for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {

			arma::vec test_destination_point = dcm_S.t() * ( dcm_D * destination_pc -> get_point_coordinates(destination_source_dist_vector[i].first) + x_D - x_S);


			int index_closest_source_point = source_pc -> get_closest_point(test_destination_point);
			arma::vec closest_source_point = source_pc -> get_point_coordinates(index_closest_source_point);

			arma::vec n_source = dcm_S * source_pc -> get_normal_coordinates(index_closest_source_point);
			arma::vec n_destination = dcm_D * destination_pc -> get_normal_coordinates(destination_source_dist_vector[i].first);

		// If the two normals are compatible, the points are matched
			if (arma::dot(n_source,n_destination) > std::sqrt(2) / 2 ) {
				destination_source_dist_vector[i].second = index_closest_source_point;
			}

		}



	// The source point is mapped to the destination frame using the inverse of the a-priori transform
	// Then, the destination pc is queried for the closest destination point
	// to the mapped source point
	// This double mapping process gets rid of edge points


	#pragma omp parallel for
		for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {
			
			if (destination_source_dist_vector[i].second != -1){
				

				arma::vec::fixed<3> test_source_point = dcm_D.t() * (dcm_S * source_pc -> get_point_coordinates(destination_source_dist_vector[i].second)+ x_S - x_D);
				int index_closest_destination_point = destination_pc -> get_closest_point(test_source_point);

				const PointNormal & closest_destination_point = destination_pc -> get_point(index_closest_destination_point);

				arma::vec::fixed<3> n_dest = dcm_D * closest_destination_point.get_normal_coordinates();
				arma::vec::fixed<3> n_source = dcm_S * source_pc -> get_normal_coordinates(destination_source_dist_vector[i].second);

		// If the two normals are compatible, the points are matched
				if (arma::dot(n_dest,n_source) > std::sqrt(2) / 2 ) {
					destination_source_dist_vector[i].first = index_closest_destination_point;

				}

			}

		}

	// The source/destination pairs are pre-formed

		for (unsigned int i = 0; i < destination_source_dist_vector.size(); ++i) {

			if (destination_source_dist_vector[i].second != -1){
				arma::vec S = dcm_S * source_pc -> get_point_coordinates(destination_source_dist_vector[i].second) + x_S;
				arma::vec n = dcm_D * destination_pc -> get_normal_coordinates(destination_source_dist_vector[i].first);
				arma::vec D = dcm_D * destination_pc -> get_point_coordinates(destination_source_dist_vector[i].first) + x_D;

				formed_pairs.push_back(std::make_pair(i,arma::dot(n,S - D)));
			}
		}	


	}



	#if ICP2P_DEBUG
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

	



	arma::gmm_diag model_residuals;
	std::set<unsigned int> acceptable_pairs;
	arma::urowvec residuals_gaus_ids;

	int N_clusters = 3;


		// Init GMM 
	#if ICP2P_DEBUG
	std::cout << "\tUsing " << N_clusters << " mixtures\n";
	#endif

	acceptable_pairs.clear();

		// Training GMM
	model_residuals.learn(abs(dist_vec).t(), N_clusters, arma::maha_dist, arma::random_subset, 10, 10, 1e-10, false);
	residuals_gaus_ids = model_residuals.assign( abs(dist_vec).t(), arma::prob_dist);

		// GMM learned parameters
	arma::urowvec hist = arma::hist(residuals_gaus_ids,arma::regspace<arma::urowvec>(0,N_clusters - 1));
	
	#if ICP2P_DEBUG
	model_residuals.means.print("\tResiduals GMM means: ");
	arma::sqrt(model_residuals.dcovs).print("\tResiduals GMM standard deviations: ");
	arma::rowvec(model_residuals.means - 3 * arma::sqrt(model_residuals.dcovs)).print("\tResiduals GMM means minus 3 standard deviations: ");
	hist.print("\tPopulation of each cluster: ");
	#endif


	// The acceptable clusters are stored
	arma::urowvec most_populated_clusters = arma::find(hist == hist.max()).t();
	double largest_acceptable_error = 1.2 * arma::min(model_residuals.means(most_populated_clusters));
	#if ICP2P_DEBUG
	std::cout<< "\t\tClustering achieved. Maximum acceptable cluster mean error: " << largest_acceptable_error << std::endl;
	#endif


	for (unsigned int p = 0; p < N_clusters; ++p){
		if (model_residuals.means(p) <= largest_acceptable_error){
			acceptable_pairs.insert(p);
		}
	}

	for (unsigned int i = 0; i < dist_vec.n_rows; ++i) {

		if (acceptable_pairs.find(residuals_gaus_ids(i)) != acceptable_pairs.end() ){

			std::cout << formed_pairs[i].first << std::endl;
			std::cout << destination_source_dist_vector.size() << std::endl;
			std::cout << destination_source_dist_vector[formed_pairs[i].first].second <<" " << destination_source_dist_vector[formed_pairs[i].first].first << std::endl << std::endl;

			point_pairs.push_back(std::make_pair(destination_source_dist_vector[formed_pairs[i].first].second,
				destination_source_dist_vector[formed_pairs[i].first].first));



		}

	}





	#if ICP2P_DEBUG
	std::cout << "\tKept " << point_pairs.size() << " pairs\n";
	#endif





















}



void IterativeClosestPointToPlane::build_matrices(
	const int pair_index,
	const arma::vec::fixed<3> & mrp, 
	const arma::vec::fixed<3> & x,
	arma::mat::fixed<6,6> & info_mat_temp,
	arma::vec::fixed<6> & normal_mat_temp,
	arma::vec & residual_vector,
	arma::vec & sigma_vector,
	const double & w,
	const double & los_noise_sd_baseline,
	const arma::mat::fixed<3,3> & M_pc_D){


	const arma::vec::fixed<3> & S_i = this -> pc_source -> get_point_coordinates(point_pairs[pair_index].first);
	const arma::vec::fixed<3> & D_i = this -> pc_destination -> get_point_coordinates(point_pairs[pair_index].second);
	const arma::vec::fixed<3> & n_i = this -> pc_destination -> get_normal_coordinates(point_pairs[pair_index].second);

	arma::vec::fixed<3> e = {1,0,0};

	// The noise statistics of each measurement is different.
	// The measurements are uncorrelated but the corresponding R matrix is not proportional to the identity (but only diagonal)
	arma::mat::fixed<3,3> dcm_S = RBK::mrp_to_dcm(mrp) ;
	double sigma_y_sq = std::pow(los_noise_sd_baseline,2) * arma::dot(n_i,
		(dcm_S * e * e.t() * dcm_S.t() 
			+ M_pc_D * e * e.t() *   M_pc_D.t()) * n_i);


	// the normal is also uncertain
	double sigma_theta = 10. * arma::datum::pi / 180.; // assume normals are known within a 30 = 3 * 10 degree cone
	arma::mat::fixed<3,3> P_n = std::pow(sigma_theta,2) / 2 * (arma::eye<arma::mat>(3,3) - n_i * n_i.t());

	sigma_y_sq += arma::dot(dcm_S * S_i + x - D_i,
		P_n * (dcm_S * S_i + x - D_i));


	// The partial derivative of the observation model is computed
	arma::rowvec::fixed<6> H = arma::zeros<arma::rowvec>(6);

	H.subvec(0,2) = - n_i.t();
	H.subvec(3,5) = - 4 * n_i.t() * RBK::tilde(dcm_S * S_i);

	info_mat_temp = H.t() * H / sigma_y_sq;

	normal_mat_temp =  H.t() * (arma::dot(n_i.t(),dcm_S * S_i + x - D_i)) / sigma_y_sq;

	residual_vector(pair_index) = arma::dot(n_i.t(),dcm_S * S_i + x - D_i);
	sigma_vector(pair_index) = std::sqrt(sigma_y_sq);


}




