

#include "boost/progress.hpp"

#include <PC.hpp>
#include <chrono>
#include <iostream>
#include <armadillo>
#include <chrono>
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <EstimateNormals.hpp>


int main() {
	arma::vec p = {0.05,0.01,-0.1};


	auto start = std::chrono::steady_clock::now();

	PointCloud<PointNormal> point_cloud("../bunny000.obj");

	auto end = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	std::cout << "took " << elapsed_seconds << " to load\n";
		
	start = std::chrono::steady_clock::now();
	point_cloud.build_kdtree();
	end = std::chrono::steady_clock::now();
	elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	std::cout << "took " << elapsed_seconds << " to build kd tree\n";


	EstimateNormals<PointCloud<PointNormal> > normal_estimator(point_cloud);
	arma::vec::fixed<3> los = {0,-1,0};
	start = std::chrono::steady_clock::now();

	normal_estimator.estimate_normals(5,los);
	end = std::chrono::steady_clock::now();

	elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	std::cout << "took " << elapsed_seconds << " to compute normals\n";
	std::cout << "Example normal: " << point_cloud.get_point(10).get_normal_coordinates().t() << "\n";



	
	return (0);
}













