#ifndef HEADER_MATCHFEATURE
#define HEADER_MATCHFEATURE


#include <PointCloud.hpp>
#include <PointDescriptor.hpp>




typedef typename std::pair<int, int > PointPair ;
typedef typename std::map<double ,std::pair< std::vector<int> , std::vector<int> > > Eset;

template <class T> class FeatureMatching{

public:

	/**
	Constructor
	@param pc1 Reference to first descriptor point cloud . The features in pc1 will be associated with their nearest neighbors in pc2
	@param pc2 Reference to second descriptor point cloud. Will have its kd tree computed/recomputed.
	@param N number of closest neighbors in feature space to search
	*/	
	FeatureMatching(const PointCloud<T> & pc1, PointCloud<T> & pc2);
	void match(std::map<int, std::vector< int > > & matches,const int & N); 
	void match(std::vector< PointPair > & matches,const int & N);


	static void greedy_pairing(
		int N_potential_correspondances, 
		int N_matches,
		const PointCloud<PointNormal> & point_pc1,
		const PointCloud<PointNormal> & point_pc2,
		const PointCloud<T> & descriptor_pc1,
		PointCloud<T> & descriptor_pc2,
		std::vector< PointPair > & matches);



	static void save_matches(std::string path,const std::vector<PointPair> & matches,
		const PointCloud<T> & input_pc1,
		const PointCloud<T> & input_pc2,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3)) ;

protected:
	const PointCloud<T> & pc1;
	PointCloud<T> & pc2;


	static double find_best_q_indices(
		const std::vector<int> & p_indices,
		std::vector<int> & q_indices, 
		const PointCloud<PointNormal> & point_pc1,
		const PointCloud<PointNormal> & point_pc2,
		const std::vector<int> & q_i,
		const std::vector<int> & q_j);


	static double dRMS(const std::vector<int> & p_indices,
		const std::vector<int> & q_indices, 
		const PointCloud<PointNormal> & point_pc1,
		const PointCloud<PointNormal> & point_pc2);


	static void merge_set(Eset & old_set,
		const PointCloud<PointNormal> & point_pc1,
		const PointCloud<PointNormal> & point_pc2);



};


#endif