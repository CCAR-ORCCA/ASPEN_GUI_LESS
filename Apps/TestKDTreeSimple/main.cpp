

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
	PointCloud<PointNormal> point_pc_1("../bunny000.obj");

	arma::vec::fixed<3> x = {5e-1,0.0,0.0};
	arma::vec::fixed<3> mrp = {-0.1,0.2,-0.3};

	// kdtree
	auto start = std::chrono::system_clock::now();
	point_pc_1.build_kdtree();
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> diff = end-start;
	std::cout <<"Time spent building tree : " << diff.count() << " s\n";


	// Neighbors query

	start = std::chrono::system_clock::now();
	std::vector<int> neighbors = point_pc_1.get_nearest_neighbors_radius(point_pc_1.get_point_coordinates(0), 1e-1);
	end = std::chrono::system_clock::now();

	diff = end-start;

	std::cout <<"Time spent querying neighbors : " << diff.count() << " s\n";
	std::cout << "Found " << neighbors.size() << std::endl;
	

	//  Normal estimation
	arma::vec::fixed<3> los_1 = {0,1,0};
		
	// Normal estimation
	EstimationNormals<PointNormal, PointNormal> normal_estimator_1(point_pc_1,point_pc_1);
	normal_estimator_1.set_los_dir(los_1);
	start = std::chrono::system_clock::now();
	normal_estimator_1.estimate(8);

	end = std::chrono::system_clock::now();

	diff = end-start;

	std::cout <<"Time spent computing normals : " << diff.count() << " s\n";

	// Feature estimation
	PointCloud<PointDescriptor> descriptor_pc_1(point_pc_1.size());
	EstimationFPFH<PointNormal,PointDescriptor > fpfh_estimator_1(point_pc_1,descriptor_pc_1);
	fpfh_estimator_1.set_scale_distance(true);
	
	start = std::chrono::system_clock::now();
	fpfh_estimator_1.estimate(5e-3);
	end = std::chrono::system_clock::now();

	diff = end-start;

	std::cout <<"Time spent computing fpfh : " << diff.count() << " s\n";




	return (0);
}













