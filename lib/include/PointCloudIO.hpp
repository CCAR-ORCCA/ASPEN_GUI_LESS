#ifndef HEADER_POINTCLOUDIO
#define HEADER_POINTCLOUDIO

#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <PointDescriptor.hpp>


template <class T> 
class PointCloudIO{

public:

	static void save_to_obj(const PointCloud<T> & pc, std::string savepath,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3), 
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3));
	static void save_to_txt(const PointCloud<T> & pc, std::string savepath,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3), 
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3));

	static void save_active_features_positions(
		const PointCloud<PointNormal> & pc_points,
		const PointCloud<PointDescriptor> & pc_features, 
		std::string savepath,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3), 
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3));

private:

};


#endif