#include <FeatureMatching.hpp>
#include <PointCloud.hpp>
#include <PointDescriptor.hpp>
#include <PointNormal.hpp>
#include <chrono>
#include <assert.h>

#define FEATURE_MATCHING_DEBUG 1

template <class T>
FeatureMatching<T>::FeatureMatching(const PointCloud<T> & pc1, PointCloud<T> & pc2) : pc1(pc1),pc2(pc2){

}

template <class T>
void FeatureMatching<T>::match(std::map<int , std::vector< int > > & matches,const int & N){

	// The kd tree in pc2 is recomputed
	this -> pc2.build_kdtree();

	std::vector<std::vector< int > > matches_temp;

	matches_temp.resize(pc1.size());
	
	// Each active features in pc1 is matched to its N closest neighbors (that are also active?) in pc2
	#pragma omp parallel for
	for (int i = 0; i < pc1.size(); ++i){
		
		// Skipping this feature if it is not active 
		if (!pc1.get_point(i).get_is_valid_feature()){
			continue;
		}

		auto closest_N_points = pc2.get_closest_N_points(pc1.get_point(i).get_histogram(),N);
		for (auto it = closest_N_points.begin(); it != closest_N_points.end(); ++it){
			matches_temp[i].push_back(it -> second);
		}
	}

	// Only the active features are kept
	for (int i = 0; i < matches_temp.size(); ++i){
		if (matches_temp[i].size()>0){
			matches[i] = std::vector< int >();
			for (auto it = matches_temp[i].begin(); it != matches_temp[i].end(); ++it){
				matches[i].push_back(*it);
			}
		}
	}

}



template <class T>
void FeatureMatching<T>::match(std::vector< PointPair >  & matches,const int & N){

	// The kd tree in pc2 is recomputed
	this -> pc2.build_kdtree();

	std::vector<std::vector< int > > matches_temp;

	matches_temp.resize(this -> pc1.size());



	// Each active features in pc1 is matched to its N closest neighbors (that are also active) in pc2
	#pragma omp parallel for
	for (int i = 0; i < this -> pc1.size(); ++i){
		// Skipping this feature if it is not active 
		if (!this -> pc1.get_point(i).get_is_valid_feature()){
			continue;
		}

		auto closest_N_points = this -> pc2.get_closest_N_points(this -> pc1.get_point(i).get_histogram(),N);
		for (auto it = closest_N_points.begin(); it != closest_N_points.end(); ++it){
			matches_temp[i].push_back(it -> second);
		}
	}


	
	// Only the active features are kept
	for (int i = 0; i < matches_temp.size(); ++i){
		if (matches_temp[i].size()>0){
			for (auto it = matches_temp[i].begin(); it != matches_temp[i].end(); ++it){
				matches.push_back(std::make_pair(i,*it));
			}
		}
	}

}



template <>
void FeatureMatching<PointNormal>::save_matches(std::string path,
	const std::vector<PointPair> & matches,
	const PointCloud<PointNormal> & input_pc1,
	const PointCloud<PointNormal> & input_pc2,
	const arma::mat::fixed<3,3> & dcm, 
	const arma::vec::fixed<3> & x){

	arma::mat matches_arma = arma::zeros<arma::mat>(matches.size(),6);
	arma::imat indices_matches_arma = arma::zeros<arma::imat>(matches.size(),2);


	for (unsigned int k  = 0; k < matches.size(); ++k){
		matches_arma.submat(k,0,k,2) = (dcm * input_pc1.get_point_coordinates(matches[k].first) + x).t();
		matches_arma.submat(k,3,k,5) = input_pc2.get_point_coordinates(matches[k].second).t();
		indices_matches_arma(k,0) = matches[k].first;
		indices_matches_arma(k,1) = matches[k].second;

	}

	matches_arma.save(path + ".txt",arma::raw_ascii);
	indices_matches_arma.save(path + "_indices.txt",arma::raw_ascii);

}




template <class T>
void FeatureMatching<T>::greedy_pairing(int N, 
	const PointCloud<PointNormal> & point_pc1,
	const PointCloud<PointNormal> & point_pc2,
	const PointCloud<T> & descriptor_pc1,
	PointCloud<T> & descriptor_pc2,
	std::vector< PointPair > & matches){


	// The kd tree in pc2 is recomputed
	descriptor_pc2.build_kdtree();
	std::vector< std::pair<int, std::vector< int > > > matches_temp,matches_active_only;
	matches_temp.resize(point_pc1.size());
	

	#if FEATURE_MATCHING_DEBUG
	std::cout << "Extracting correspondances...\n";
	#endif

	// Each active features in pc1 is matched to its N closest neighbors (that are also active) in pc2
	#pragma omp parallel for
	for (int i = 0; i < descriptor_pc1.size(); ++i){
		// Skipping this feature if it is not active 
		if (!descriptor_pc1.get_point(i).get_is_valid_feature()){
			continue;
		}

		auto closest_N_points = descriptor_pc2.get_closest_N_points(descriptor_pc1.get_point(i).get_histogram(),N);
		for (auto it = closest_N_points.begin(); it != closest_N_points.end(); ++it){
			matches_temp[i].first = i;
			matches_temp[i].second.push_back(it -> second);
		}
	}

	// Only working with N matches
	int N_matches = 200;

	arma::ivec random_indices = arma::shuffle(arma::regspace<arma::ivec>(0,matches_temp.size() - 1));

	for (int i = 0; i < matches_temp.size(); ++i){
		if (matches_temp[random_indices(i)].second.size()> 0){
			matches_active_only.push_back(matches_temp[random_indices(i)]);
		}
		if (matches_active_only.size() == N_matches){
			break;
		}

	}

	assert(matches_active_only.size() == N_matches);

	#if FEATURE_MATCHING_DEBUG
	std::cout << "Kept " << matches_active_only.size() <<  " active features in pc1 \n";
	#endif


	// matches_temp now contains { ... <pi|qi_1,qi_2,...,qi_N> ...  } , i = 1 .. m . Somes pi's have no correspondances because 
	// they were not valid features to begin with

	// Getting the sorted set of two-point correspondances {... <pi,pj|qi,qj> ... }
	Eset E2;

	#if FEATURE_MATCHING_DEBUG
	std::cout << "Getting the sorted set of two-point correspondances {... <pi,pj|qi,qj> ... } ...\n";
	#endif

	int combinations = (N_matches - 1) + static_cast<int>((N_matches - 1) * (N_matches - 2)/2.);
	std::vector<double> errors(combinations);
	std::vector< std::pair< std::vector<int> , std::vector<int> > >  p_q_indices(combinations);


	auto start = std::chrono::system_clock::now();
	#pragma omp parallel for
	for (int i = 0; i < N_matches; ++i){
		for (int j = 0; j < i; ++j){
			int global_index = j + static_cast<int>(i * (i - 1)/2.);

			// we have a (pi,pj) pair of valid features on the source point cloud. 
			// Must now browse their correspondances to find (qi,qj) minimizing the RMS distance
			std::vector<int> p_indices = {
				matches_active_only[i].first,
				matches_active_only[j].first
			};

			std::vector<int> q_indices(2);
			const std::vector<int> & q_i = matches_active_only[i].second;
			const std::vector<int> & q_j = matches_active_only[j].second;

			// The q_indices must come from 
			// matches_active_only[i].second, and
			// matches_active_only[j].second

			double error = FeatureMatching<T>::find_best_q_indices(p_indices,q_indices,point_pc1,point_pc2,q_i,q_j);
			errors[global_index] = error;
			p_q_indices[global_index] = std::make_pair(p_indices,q_indices);

		}
	}


	#if FEATURE_MATCHING_DEBUG
	std::vector< PointPair > matches_before;
	for (auto it = p_q_indices.begin(); it != p_q_indices.end(); ++it){
		for (int i = 0; i < it -> first.size(); ++i){
			matches_before.push_back(std::make_pair(it -> first[i],it -> second[i]));			
		}	
	}

	FeatureMatching<PointNormal>::save_matches("matches_before",matches_before,point_pc1,point_pc2);

	#endif



	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end-start;
	std::cout <<"Time elapsed building E2 vector: " << diff.count() << " s\n";

	// Filling out the map
	start = std::chrono::system_clock::now();
	for (int global_index = 0; global_index < combinations; ++global_index){
		E2[errors[global_index]] = p_q_indices[global_index];
	}
	end = std::chrono::system_clock::now();
	diff = end-start;
	std::cout << "Time elapsed building E2 map: " << diff.count() << " s\n";


	// The sets are merged 
	for (int n = 0; n < 5; ++n){
		#if FEATURE_MATCHING_DEBUG
		std::cout << "Current level : " << n << "\n";
		#endif
		FeatureMatching<T>::merge_set(E2,point_pc1,point_pc2);
	}


	// Finally storing matches from the best transform
	auto it = E2.begin();
	for (int i = 0; i < it -> second.first.size(); ++i){
		matches.push_back(std::make_pair(it -> second.first[i],it -> second.second[i]));
	}	
	


}



template <class T>
double FeatureMatching<T>::find_best_q_indices(
	const std::vector<int> & p_indices,
	std::vector<int> & q_indices, 
	const PointCloud<PointNormal> & point_pc1,
	const PointCloud<PointNormal> & point_pc2,
	const std::vector<int> & q_i,
	const std::vector<int> & q_j){


	// Given the p_indices (there are K of them) , must find the K q_indices amongst 
	// their potential correspondances than minimizes their dRMS
	// for instance, in E2, p_indices = <i,j>. So if each point has 10 potential correspondances,
	// we have a total of 100 cases to check
	std::vector<int> q_indices_temp(2);
	double best_dRMS = std::numeric_limits<double>::infinity();


	for (int k = 0; k < q_i.size(); ++k){

		for (int l = 0; l < q_j.size(); ++l){

			q_indices_temp[0] = q_i[k];
			q_indices_temp[1] = q_j[l];

			double temp_dRMS = dRMS(p_indices,q_indices_temp, point_pc1,point_pc2);

			if (temp_dRMS < best_dRMS){
				best_dRMS = temp_dRMS;
				q_indices = q_indices_temp;
			}


		}

	}
	return best_dRMS;

}


template <class T>
double FeatureMatching<T>::dRMS(
	const std::vector<int> & p_indices,
	const std::vector<int> & q_indices, 
	const PointCloud<PointNormal> & point_pc1,
	const PointCloud<PointNormal> & point_pc2){

	double dRMS = 0;

	for (int i = 0; i < p_indices.size(); ++i){
		for (int j = 0; j < p_indices.size(); ++j){

			dRMS += std::pow(
				arma::norm(point_pc1.get_point_coordinates(p_indices[i]) - point_pc1.get_point_coordinates(p_indices[j])) 
				- arma::norm(point_pc2.get_point_coordinates(q_indices[i]) - point_pc2.get_point_coordinates(q_indices[j])),2);
		}
	}


	return dRMS / std::pow(p_indices.size(),2);


}


template <class T>
void  FeatureMatching<T>::merge_set(Eset & old_set,
	const PointCloud<PointNormal> & point_pc1,
	const PointCloud<PointNormal> & point_pc2){

	Eset new_set;

	while(old_set.size() > 1){

		double best_error = std::numeric_limits<double>::infinity();
		std::vector<int> best_p_indices,best_q_indices;

		// For a given N-point correspondance in old_set,
		// find the N-point correspondance that best completes it
		// We are prioritizing lower-error pairs since the map is ordered

		#if FEATURE_MATCHING_DEBUG
		std::cout << "Merging ... (have found " <<new_set.size() << " next-level correspondances so far) \n";
		#endif
		auto e = old_set.begin();

		for (auto second_cor_it = old_set.begin(); second_cor_it != old_set.end(); ++second_cor_it){

			if (second_cor_it == e){
				continue;
			}

			std::vector<int> p_indices = e -> second.first;
			p_indices.insert(p_indices.end(),second_cor_it -> second.first.begin(),second_cor_it -> second.first.end());


			std::vector<int> q_indices = e -> second.second;
			q_indices.insert(q_indices.end(),second_cor_it -> second.second.begin(),second_cor_it -> second.second.end());

			double error = dRMS(p_indices,q_indices, point_pc1,point_pc2);
			if (error < best_error){
				best_error = error;
				best_p_indices = p_indices;
				best_q_indices = q_indices;
			}

		}

		std::cout << "Best error: " << best_error << std::endl;
		for (int i = 0 ; i < best_p_indices.size(); ++i){
			std::cout << "p_" << i << ": " <<   best_p_indices[i] << "," << "q_" << i << ": "  << best_q_indices[i]  << std::endl;
		}
		std::cout << std::endl; 



		// At this stage, we have found the best correspondance *second_cor_it that completes *e
		new_set[best_error] = std::make_pair(best_p_indices,best_q_indices);

		// The old set is now purged from correspondances whose endpoints union give best_q_indices
		// For each entry in the old set
		#if FEATURE_MATCHING_DEBUG
		std::cout << "Purging old_set: " << old_set.size() << "\n";
		#endif

		for (auto it = old_set.begin(); it != old_set.end(); ){

			bool must_erase = false;

			// The q in that correspondance must 
			for (int i = 0; i < it -> second.second.size(); ++i){

				if (std::find(best_q_indices.begin(),best_q_indices.end(),it -> second.second[i]) != best_q_indices.end()){
					// This q index in old set has already been added to the new set
					must_erase = true;
					break;
				}

			}

			if (must_erase){
				it = old_set.erase(it);
				#if FEATURE_MATCHING_DEBUG
				std::cout << "\tElements left in old_set: " << old_set.size() << "\n";
			#endif
			}
			else{
				++it;
			}

			
		}

	}

	if (new_set.size() > 0)
		old_set = new_set;

	#if FEATURE_MATCHING_DEBUG
	std::cout << "Found " << old_set.size() << " correspondances now featuring " << old_set.begin() -> second.first.size() << " points \n";
	std::cout << "Smallest error: " << old_set.begin()-> first << std::endl;
	#endif






}










template class FeatureMatching<PointDescriptor> ;
