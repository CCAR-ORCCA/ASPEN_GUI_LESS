

#include "boost/progress.hpp"

#include <chrono>
#include <iostream>
#include <armadillo>
#include <chrono>
#include <PointCloud.hpp>
#include <PointCloudIO.hpp>

#include <PointNormal.hpp>
#include <EstimationNormals.hpp>
#include <EstimationPFH.hpp>
#include <EstimationFPFH.hpp>
#include <FeatureMatching.hpp>

#include <IterativeClosestPoint.hpp>
#include <IterativeClosestPointToPlane.hpp>



int main() {

	// loading
	PointCloud<PointNormal> input_pc_1("../bunny000.obj");
	PointCloud<PointNormal> input_pc_2("../bunny090.obj");

	arma::vec::fixed<3> x = {5e-1,0.0,0.0};
	arma::vec::fixed<3> mrp = {-0.1,0.2,-0.3};

	// kdtree
	input_pc_1.build_kdtree();
	input_pc_2.build_kdtree();

	//  Normal estimation
	arma::vec::fixed<3> los_1 = {0,1,0};
	arma::vec::fixed<3> los_2 = {0,0,1};

	EstimationNormals<PointNormal, PointNormal> normal_estimator_1(input_pc_1,input_pc_1);
	EstimationNormals<PointNormal, PointNormal> normal_estimator_2(input_pc_2,input_pc_2);
	normal_estimator_1.set_los_dir(los_1);
	normal_estimator_2.set_los_dir(los_2);

	normal_estimator_1.estimate(8);
	normal_estimator_2.estimate(8);
	
	// input_pc_2.transform(RBK::mrp_to_dcm(mrp),x);
	// input_pc_2.build_kdtree();

	std::cout << "computing features\n";

	//  FPFH estimation
	PointCloud<PointDescriptor> output_pc_1(input_pc_1.size());
	PointCloud<PointDescriptor> output_pc_2(input_pc_2.size());

	EstimationFPFH<PointNormal,PointDescriptor > fpfh_estimator_1(input_pc_1,output_pc_1);
	fpfh_estimator_1.set_scale_distance(true);

	EstimationFPFH<PointNormal,PointDescriptor > fpfh_estimator_2(input_pc_2,output_pc_2);
	fpfh_estimator_2.set_scale_distance(true);

	fpfh_estimator_1.estimate(5e-3);
	fpfh_estimator_2.estimate(5e-3);
	fpfh_estimator_1.prune(1.25);
	fpfh_estimator_2.prune(1.25);

	PointCloudIO<PointDescriptor>::save_to_txt(output_pc_1,"fpfh_1.txt");
	PointCloudIO<PointDescriptor>::save_to_txt(output_pc_2,"fpfh_2.txt");

	std::cout << "matching features\n";
	
	// FPFH matching
	FeatureMatching<PointDescriptor> feature_matching(output_pc_1,output_pc_2);
	std::vector< PointPair > matches;
	feature_matching.match(matches,2);
	FeatureMatching<PointNormal>::save_matches("all_matches.txt",matches,input_pc_1,input_pc_2);


	PointCloudIO<PointNormal>::save_to_obj(input_pc_1,"input_pc_1.obj");
	PointCloudIO<PointNormal>::save_to_obj(input_pc_2,"input_pc_2.obj");
	PointCloudIO<PointNormal>::save_to_txt(input_pc_1,"input_pc_1.txt");
	PointCloudIO<PointNormal>::save_to_txt(input_pc_2,"input_pc_2.txt");

	std::shared_ptr<PointCloud<PointNormal > > input_pc_1_ptr = std::make_shared<PointCloud<PointNormal > >(input_pc_1);
	std::shared_ptr<PointCloud<PointNormal > > input_pc_2_ptr = std::make_shared<PointCloud<PointNormal > >(input_pc_2);

	arma::mat::fixed<3,3> dcm_ransac;
	arma::vec::fixed<3> x_ransac;
	std::vector< PointPair > matches_ransac;

	IterativeClosestPointToPlane::ransac(
		matches,
		20,
		100,
		1e-2,
		100,
		input_pc_1_ptr,
		input_pc_2_ptr,
		dcm_ransac,
		x_ransac,
		matches_ransac);

	PointCloudIO<PointNormal>::save_to_obj(input_pc_1,"registered_pc_1.obj",dcm_ransac,x_ransac);
	FeatureMatching<PointNormal>::save_matches("matches_ransac.txt",matches_ransac,input_pc_1,input_pc_2);




	PointCloud<PointNormal> active_features_pc_1,active_features_pc_2;
	
	for (int i = 0; i < input_pc_1.size(); ++i){
		if (output_pc_1[i].get_is_valid_feature()){
			active_features_pc_1.push_back(input_pc_1[i]);
		}
	}

	for (int i = 0; i < input_pc_2.size(); ++i){
		if (output_pc_2[i].get_is_valid_feature()){
			active_features_pc_2.push_back(input_pc_2[i]);
		}
	}

	PointCloudIO<PointNormal>::save_to_obj(active_features_pc_1,"active_features_pc_1.obj");
	PointCloudIO<PointNormal>::save_to_obj(active_features_pc_2,"active_features_pc_2.obj");



	return (0);
}













