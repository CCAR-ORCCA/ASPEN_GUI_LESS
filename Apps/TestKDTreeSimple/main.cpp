

#include "boost/progress.hpp"

#include <PC.hpp>
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


int main() {

	double radius = 7e-3;

	// loading
	auto start = std::chrono::steady_clock::now();
	PointCloud<PointNormal> point_cloud("../bunny000.obj");
	std::cout << "Point cloud has " << point_cloud.size() << " points\n";
	auto end = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	std::cout << "took " << elapsed_seconds << " to load\n";
	
	// kdtree
	start = std::chrono::steady_clock::now();
	point_cloud.build_kdtree();
	end = std::chrono::steady_clock::now();
	elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	std::cout << "took " << elapsed_seconds << " to build kd tree\n";

	// Nearest neighbors of query point

	const arma::vec & p = point_cloud.get_point_coordinates(0);


	start = std::chrono::steady_clock::now();
	auto neighbors = point_cloud.get_nearest_neighbors_radius(p, radius);
	end = std::chrono::steady_clock::now();
	elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	std::cout << "Found " << neighbors.size() << " neighbors in " << elapsed_seconds << " s\n";


	//  Normal estimation
	EstimationNormals<PointCloud<PointNormal> > normal_estimator(point_cloud);
	arma::vec::fixed<3> los = {0,-1,0};
	start = std::chrono::steady_clock::now();
	normal_estimator.estimate_normals(radius,los);
	end = std::chrono::steady_clock::now();
	elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	std::cout << "took " << elapsed_seconds << " to compute normals\n";


	//  FPFH estimation
	EstimationFPFH<PointCloud<PointNormal>,PointCloud<PointDescriptor> > fpfh_estimator(point_cloud);
	PointCloud<PointDescriptor> output_pc(point_cloud.size());
	fpfh_estimator.set_scale_distance(true);
	fpfh_estimator.estimate(2 * radius,11,output_pc);

	PointCloudIO<PointNormal>::save_to_obj(point_cloud,"test_point_cloud.obj");
	PointCloudIO<PointNormal>::save_to_txt(point_cloud,"test_point_cloud.txt");
	PointCloudIO<PointDescriptor>::save_to_txt(output_pc,"test_fpfh_cloud.txt");


	
	return (0);
}













