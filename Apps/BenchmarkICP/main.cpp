
#include "PC.hpp"
#include "ICP.hpp"
#include "IterativeClosestPointToPlane.hpp"
#include "IterativeClosestPoint.hpp"
#include <armadillo>
#include "boost/progress.hpp"
#include <ShapeModelImporter.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>
#include <chrono>

#include <iostream>
// #include <pcl/io/pcd_io.h>
// #include <pcl/io/obj_io.h>

// #include <pcl/registration/icp.h>
// #include <pcl/registration/gicp.h>
#include <RigidBodyKinematics.hpp>
#include <PointDescriptor.hpp>

int main() {

	// Perturbed source pc
	PC destination_pc("../bunny000.obj");
	// PC source_pc("../bunny090.obj");

	arma::vec los = {0,-1,0};


	std::cout << "Constructing normals...\n";
	auto start = std::chrono::steady_clock::now();
	destination_pc.construct_normals(los,7e-3);
	auto end = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<
	std::chrono::duration<double> >(finish - start).count();
	std::cout << "Normals were computed in " << elapsed_seconds << " seconds\n";

	std::cout << "Computing neighborhoods...\n";

	start = std::chrono::steady_clock::now();

	destination_pc.compute_neighborhoods(14e-3);

	end = std::chrono::steady_clock::now();

	elapsed_seconds = std::chrono::duration_cast<
	std::cout << "Neighborhoods were computed in " << elapsed_seconds << " seconds\n";


	std::cout << "Computing FPFH...\n";

	destination_pc.compute_FPFH(false,11,14e-3);

	std::cout << "Saving features...\n";

	destination_pc.save_active_features(0,"bunny");

	// destination_pc.compute_feature_descriptors(PC::FeatureDescriptor::FPFHDescriptor,false,11,14e-3,"bunny");

	throw;




	// // ASPEN ICP
	// std::shared_ptr<PC> destination_pc_ptr = std::make_shared<PC>(destination_pc);
	// std::shared_ptr<PC> source_pc_ptr = std::make_shared<PC>(source_pc);

	// IterativeClosestPoint icp_ransac(destination_pc_ptr, source_pc_ptr);
	// icp_ransac.set_keep_correlations(false);
	// icp_ransac.set_N_bins(11);
	// icp_ransac.set_neighborhood_radius(3e-3);
	// icp_ransac.set_use_FPFH(true);


	// icp_ransac.register_pc_bf(1000,3,30);

	// source_pc_ptr -> save("../source_solution_aspen_ransac.obj",icp_ransac.get_dcm(),icp_ransac.get_x());
	// IterativeClosestPointToPlane icp(destination_pc_ptr, source_pc_ptr);
	// icp.register_pc(icp_ransac.get_dcm(),icp_ransac.get_x());
	// source_pc_ptr -> save("../source_solution_aspen_final.obj",icp.get_dcm(),icp.get_x());
	
	return (0);
}













