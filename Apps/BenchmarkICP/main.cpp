

#include "boost/progress.hpp"

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
	PointCloud<PointNormal> point_pc_1("../bunny270.obj");
	// PointCloud<PointNormal> point_pc_2("../bunny000.obj");

	PointCloud<PointNormal> point_pc_2("../bunny180.obj");

	arma::vec::fixed<3> x = {5e-1,0.0,0.0};
	arma::vec::fixed<3> mrp = {-0.1,0.2,-0.3};

	// kdtree
	point_pc_1.build_kdtree();
	point_pc_2.build_kdtree();

	//  Normal estimation
	arma::vec::fixed<3> los_1 = {0,0,1};
	// arma::vec::fixed<3> los_2 = {0,1,0};
	arma::vec::fixed<3> los_2 = {0,0,1};



	EstimationNormals<PointNormal, PointNormal> normal_estimator_1(point_pc_1,point_pc_1);
	EstimationNormals<PointNormal, PointNormal> normal_estimator_2(point_pc_2,point_pc_2);
	normal_estimator_1.set_los_dir(los_1);
	normal_estimator_2.set_los_dir(los_2);

	normal_estimator_1.estimate(8);
	normal_estimator_2.estimate(8);
	
	// point_pc_2.transform(RBK::mrp_to_dcm(mrp),x);
	// point_pc_2.build_kdtree();

	std::cout << "computing features\n";

	//  FPFH estimation
	PointCloud<PointDescriptor> descriptor_pc_1(point_pc_1.size());
	PointCloud<PointDescriptor> descriptor_pc_2(point_pc_2.size());

	EstimationFPFH<PointNormal,PointDescriptor > fpfh_estimator_1(point_pc_1,descriptor_pc_1);
	fpfh_estimator_1.set_scale_distance(true);

	EstimationFPFH<PointNormal,PointDescriptor > fpfh_estimator_2(point_pc_2,descriptor_pc_2);
	fpfh_estimator_2.set_scale_distance(true);

	fpfh_estimator_1.estimate(2.5e-3);
	fpfh_estimator_2.estimate(2.5e-3);
	fpfh_estimator_1.prune(1);
	fpfh_estimator_2.prune(1);

	fpfh_estimator_1.estimate(1e-2);
	fpfh_estimator_2.estimate(1e-2);
	fpfh_estimator_1.prune(1.);
	fpfh_estimator_2.prune(1.);

	// fpfh_estimator_1.estimate(7.5e-3);
	// fpfh_estimator_2.estimate(7.5e-3);
	// fpfh_estimator_1.prune(1.);
	// fpfh_estimator_2.prune(1.);

	PointCloudIO<PointNormal>::save_active_features_positions(point_pc_1,descriptor_pc_1, 
		"active_features_pc_1.obj");
	PointCloudIO<PointNormal>::save_active_features_positions(point_pc_2,descriptor_pc_2, 
		"active_features_pc_2.obj");


	PointCloudIO<PointDescriptor>::save_to_txt(descriptor_pc_1,"fpfh_1.txt");
	PointCloudIO<PointDescriptor>::save_to_txt(descriptor_pc_2,"fpfh_2.txt");

	std::cout << "matching features\n";


	PointCloudIO<PointNormal>::save_to_obj(point_pc_1,"point_pc_1.obj");
	PointCloudIO<PointNormal>::save_to_obj(point_pc_2,"point_pc_2.obj");


	std::vector< PointPair > matches;
	FeatureMatching<PointDescriptor>::greedy_pairing(5, 
	point_pc_1,
	point_pc_2,
	descriptor_pc_1,
	descriptor_pc_2,
	matches);

	FeatureMatching<PointNormal>::save_matches("all_matches",matches,point_pc_1,point_pc_2);

	std::shared_ptr<PointCloud<PointNormal > > point_pc_1_ptr = std::make_shared<PointCloud<PointNormal > >(point_pc_1);
	std::shared_ptr<PointCloud<PointNormal > > point_pc_2_ptr = std::make_shared<PointCloud<PointNormal > >(point_pc_2);

	std::cout << "Running ICP\n";
	IterativeClosestPoint icp;
	icp.set_pc_source(point_pc_1_ptr);
	icp.set_pc_destination(point_pc_2_ptr);
	icp.set_pairs(matches);
	icp.register_pc();
	PointCloudIO<PointNormal>::save_to_obj(point_pc_1,"apriori_registered_pc_1.obj",icp.get_dcm(),icp.get_x());

	IterativeClosestPointToPlane icp2p;
	icp2p.set_pc_source(point_pc_1_ptr);
	icp2p.set_pc_destination(point_pc_2_ptr);
	icp2p.set_pairs(std::vector<PointPair>());
	icp2p.register_pc(icp.get_dcm(), icp.get_x());

	PointCloudIO<PointNormal>::save_to_obj(point_pc_1,"registered_pc_1.obj",icp2p.get_dcm(),icp2p.get_x());

	return (0);
}













