
#include "PC.hpp"
#include "ICP.hpp"
#include "IterativeClosestPointToPlane.hpp"
#include "IterativeClosestPoint.hpp"
#include <armadillo>
#include "boost/progress.hpp"
#include <ShapeModelImporter.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>

#include <iostream>
// #include <pcl/io/pcd_io.h>
// #include <pcl/io/obj_io.h>

// #include <pcl/registration/icp.h>
// #include <pcl/registration/gicp.h>
#include <RigidBodyKinematics.hpp>

int main() {


	
	// Perturbed source pc

	PC destination_pc("../itokawa_64_scaled_aligned.obj");
	PC source_pc("../source_3_before.obj");



	// ASPEN ICP
	std::shared_ptr<PC> destination_pc_ptr = std::make_shared<PC>(destination_pc);
	std::shared_ptr<PC> source_pc_ptr = std::make_shared<PC>(source_pc);

	IterativeClosestPoint icp_ransac(destination_pc_ptr, source_pc_ptr);
	icp_ransac.set_keep_correlations(false);
	icp_ransac.set_N_bins(11);
	icp_ransac.set_neighborhood_radius(3e0);
	icp_ransac.set_use_FPFH(true);


	icp_ransac.register_pc_bf(3000,5,10);

	source_pc_ptr -> save("../source_solution_aspen_ransac.obj",icp_ransac.get_dcm(),icp_ransac.get_x());
	IterativeClosestPointToPlane icp(destination_pc_ptr, source_pc_ptr);
	icp.register_pc(icp_ransac.get_dcm(),icp_ransac.get_x());
	source_pc_ptr -> save("../source_solution_aspen_final.obj",icp.get_dcm(),icp.get_x());
	
	return (0);
}













