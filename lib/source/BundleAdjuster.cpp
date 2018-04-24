#include "BundleAdjuster.hpp"
#include <armadillo>
#include "ICP.hpp"
#include "boost/progress.hpp"



BundleAdjuster::BundleAdjuster(std::vector< std::shared_ptr<PC> > * all_registered_pc_){

	this -> all_registered_pc = all_registered_pc_;

	// The connectivity between point clouds is inferred
	this -> find_point_cloud_connectivity();

	// solve the bundle adjustment problem
	this -> solve_bundle_adjustment();

	// The connectivity matrix is saved
	this -> save_connectivity_matrix();
}


void BundleAdjuster::solve_bundle_adjustment(){


	int N_iter = 5;
	int Q = this -> all_registered_pc -> size();


	arma::vec dX;



	for (int iter = 0 ; iter < N_iter; ++iter){

		arma::sp_mat Lambda(6 * (Q-1),6 * (Q-1));
		arma::vec N(6 * (Q-1));

		// For each point-cloud pair
		for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){

			arma::mat Lambda_k;
			arma::vec N_k;

			// The Lambda_k and N_k specific to this point-cloud pair are computed
			this -> assemble_subproblem(Lambda_k,N_k,this -> point_cloud_pairs . at(k));

			// They are added to the whole problem
			this -> add_subproblem_to_problem(Lambda,N,Lambda_k,N_k,this -> point_cloud_pairs . at(k));

		}

		// The deviation in all of the rigid transforms is computed
		dX = arma::spsolve(Lambda,N);

		// It is applied to all of the point clouds (minus the first one)
		this -> apply_deviation(dX);

		// The point cloud pairs are updated: their residuals are update
		this -> update_point_cloud_pairs();

	}


}

void BundleAdjuster::find_point_cloud_connectivity(){

	int M = this -> all_registered_pc -> size();

	boost::progress_display progress(M - 1);

	std::vector< BundleAdjuster::PointCloudPair > point_cloud_pairs_temps;
	for (int i = 0; i < M ; ++i){
		for (int j = 0; j < i; ++j){
			point_cloud_pairs_temps.push_back(BundleAdjuster::PointCloudPair());
		}
	}


	// The point clouds connectivity is reconstructed
	#pragma omp parallel for
	for (int i = 0; i < M ; ++i){
		for (int j = 0; j < i; ++j){

			std::vector<PointPair> point_pairs;

		// Point pairs are formed between the two point clouds so as to determine what is 
		// their alignement error

			try{

				int h = 5;

				ICP::compute_pairs(point_pairs,this -> all_registered_pc -> at(i),this -> all_registered_pc -> at(j),h);				
				double error = ICP::compute_rms_residuals(point_pairs);

				double p = std::log2(this -> all_registered_pc -> at(i) -> get_size());
				int N_pairs = (int)(std::pow(2, p - h));

				BundleAdjuster::PointCloudPair pair;
				pair.S_k = i;
				pair.D_k = j;
				pair.error = error;
				pair.N_pairs = N_pairs;
				pair.N_accepted_pairs = point_pairs.size();

				point_cloud_pairs_temps[j + i * (i - 1) / 2] = pair;


			}
			catch(ICPNoPairsException & e){

			}
			catch(ICPException & e){

			}
		}
		++progress;
	}

	// The potential point-cloud pairs are formed
	for (unsigned int i = 0; i < point_cloud_pairs_temps.size(); ++i){
		if (point_cloud_pairs_temps[i].S_k >= 0){
			this -> point_cloud_pairs.push_back(point_cloud_pairs_temps[i]);
		}
	}

}



void BundleAdjuster::assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,const PointCloudPair & point_cloud_pair){

	// The point-pairs in the prescribed point-cloud pair are formed (with h = 0, so we are using them all)
	std::vector<PointPair> point_pairs;

	// The point pairs must be computed using the current estimate of the point clouds' rigid transform
	ICP::compute_pairs(point_pairs,this -> all_registered_pc -> at(point_cloud_pair.S_k),this -> all_registered_pc -> at(point_cloud_pair.D_k),0);		

	arma::rowvec H_ki = arma::zeros<arma::rowvec>(12);

	// For all the point pairs that where formed
	for (unsigned int i = 0; i < point_pairs.size(); ++i){

		double y_ki = ICP::compute_normal_distance(point_pairs[i]);

		H_ki.subvec(0,2) = point_pairs[i].second -> get_normal().t();
		H_ki.subvec(3,5) = - 4 * H_ki.subvec(0,2) * RBK::tilde(point_pairs[i].first -> get_point());
		H_ki.subvec(6,8) = - H_ki.subvec(0,2);
		H_ki.subvec(9,11) = 4 * ( H_ki.subvec(0,2) * RBK::tilde(point_pairs[i].second -> get_point()));




		Lambda_k += H_ki.t() * H_ki;
		N_k += H_ki.t() * y_ki;

	}

}

void BundleAdjuster::update_point_cloud_pairs(){


}






void BundleAdjuster::add_subproblem_to_problem(arma::sp_mat & Lambda,arma::vec & N,const arma::mat & Lambda_k,const arma::mat & N_k,
	const PointCloudPair & point_cloud_pair){

}

void BundleAdjuster::apply_deviation(const arma::vec & dX){


	for (unsigned int i = 1; i < this -> all_registered_pc -> size(); ++i){


		this -> all_registered_pc -> at(i) -> transform(RBK::mrp_to_dcm(dX.subvec(6 * (i - 1) + 3, 6 * (i - 1) + 5)), 
			dX.subvec(6 * (i - 1) , 6 * (i - 1) + 2));

	}

}



void BundleAdjuster::save_connectivity_matrix() const{
	int M = this -> point_cloud_pairs. size();
	int Q = this -> all_registered_pc -> size();

	arma::mat connectivity_matrix_res(Q,Q);
	arma::mat connectivity_matrix_overlap(Q,Q);
	arma::mat connectivity_matrix_N_pairs(Q,Q);


	connectivity_matrix_res.fill(arma::datum::nan);
	connectivity_matrix_overlap.fill(arma::datum::nan);


	for (int k = 0; k < M; ++k){
		auto point_cloud_pair = this -> point_cloud_pairs.at(k);
		connectivity_matrix_res(point_cloud_pair.S_k,point_cloud_pair.D_k) = point_cloud_pair.error;
		connectivity_matrix_res(point_cloud_pair.D_k,point_cloud_pair.S_k) = point_cloud_pair.error;

		connectivity_matrix_overlap(point_cloud_pair.S_k,point_cloud_pair.D_k) = double(point_cloud_pair.N_accepted_pairs) / point_cloud_pair.N_pairs;
		connectivity_matrix_overlap(point_cloud_pair.D_k,point_cloud_pair.S_k) = double(point_cloud_pair.N_accepted_pairs) / point_cloud_pair.N_pairs;

		connectivity_matrix_N_pairs(point_cloud_pair.S_k,point_cloud_pair.D_k) = point_cloud_pair.N_pairs;
		connectivity_matrix_N_pairs(point_cloud_pair.D_k,point_cloud_pair.S_k) = point_cloud_pair.N_pairs;

	}

	connectivity_matrix_res.save("../output/connectivity_res.csv",arma::csv_ascii);
	connectivity_matrix_overlap.save("../output/connectivity_overlap.csv",arma::csv_ascii);
	connectivity_matrix_N_pairs.save("../output/connectivity_N_pairs.csv",arma::csv_ascii);


}

