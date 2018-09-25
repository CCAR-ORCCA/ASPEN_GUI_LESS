#include <FeatureMatching.hpp>
#include <PointCloud.hpp>
#include <PointDescriptor.hpp>
#include <PointNormal.hpp>

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

	matches_temp.resize(pc1.size());
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


	std::cout << "storing matches\n";
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
	const PointCloud<PointNormal> & input_pc2){

	arma::mat matches_arma = arma::zeros<arma::mat>(matches.size(),6);

	for (unsigned int k  = 0; k < matches.size(); ++k){
		matches_arma.submat(k,0,k,2) = input_pc1.get_point_coordinates(matches[k].first).t();
		matches_arma.submat(k,3,k,5) = input_pc2.get_point_coordinates(matches[k].second).t();
	}

	matches_arma.save(path,arma::raw_ascii);

}










template class FeatureMatching<PointDescriptor> ;
