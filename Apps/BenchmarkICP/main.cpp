
#include "PC.hpp"
#include "ICP.hpp"
#include <armadillo>
#include "boost/progress.hpp"
#include <ShapeModelImporter.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <RigidBodyKinematics.hpp>

int main() {


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target (new pcl::PointCloud<pcl::PointXYZ>);

	// Destination pcs
	PC pc0("../source_0.obj");
	pcl::io::loadOBJFile ( "../source_0.obj",*cloud_target ) ;

	std::cout << "Destination pc size: " << pc0.get_size() << std::endl;

	// Perturbed source pc
	arma::vec x = {10,20,30};
	arma::vec mrp = {0.30,0.15,0.15};
	arma::mat dcm = RBK::mrp_to_dcm(mrp);
	pc0.save("../source_misaligned.obj",dcm,x);
	PC pc1("../source_misaligned.obj");
	pcl::io::loadOBJFile ( "../source_misaligned.obj",*cloud_in ) ;

	std::cout << "Source pc size: " << pc1.get_size() << std::endl;

	// PCL ICP
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	pcl::PointCloud<pcl::PointXYZ> Final;
	icp.setInputSource(cloud_in);
	icp.setInputTarget(cloud_target);
	icp.setMaximumIterations (100);
	icp.align(Final);

	// ASPEN ICP
	std::shared_ptr<PC> pc0_ptr = std::make_shared<PC>(pc0);
	std::shared_ptr<PC> pc1_ptr = std::make_shared<PC>(pc1);
	ICP icp_aspen(pc0_ptr, pc1_ptr);
	// icp_aspen.set_use_true_pairs(true);
	// icp_aspen.register_pc_mrp_multiplicative_partials(100,1e-8,1e-2);
	// pc1_ptr -> save("../source_solution_aspen.obj",icp_aspen.get_M(),icp_aspen.get_X());

	arma::mat points_transformed(3,Final.size());

	for (int i = 0; i < Final.size() ;++i){
		points_transformed(0,i) = Final.points[i].x;
		points_transformed(1,i) = Final.points[i].y;
		points_transformed(2,i) = Final.points[i].z;
	}

	arma::vec u = {1,0,0};
	PC pc_final(u,points_transformed);
	pc_final.save("../source_solution_pcl.obj");

	std::cout << "PCL solution:\n";
	std::cout <<  icp.getFinalTransformation() << std::endl;
	std::cout << "\t score: " << icp.getFitnessScore() << std::endl;
	std::cout << "ASPEN solution:\n";
	std::cout << icp_aspen.get_X() << std::endl;
	std::cout << icp_aspen.get_M() << std::endl;
	std::cout << "\t score: " << icp_aspen.get_J_res() << std::endl;



	return (0);
}













