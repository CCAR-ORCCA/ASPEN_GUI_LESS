#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelImporter.hpp>
#include <Lidar.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <RigidBodyKinematics.hpp>
#include <BundleAdjuster.hpp>
#include <EstimationNormals.hpp>
#include <IterativeClosestPointToPlane.hpp>

#include <chrono>


int main(){

	std::shared_ptr<PointCloud<PointNormal>> pc0 = std::make_shared<PointCloud<PointNormal>>(PointCloud<PointNormal>("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/destination_12.obj"))	;
	std::shared_ptr<PointCloud<PointNormal>> pc1 = std::make_shared<PointCloud<PointNormal>>(PointCloud<PointNormal>("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/destination_57_ba.obj"))	;

	pc0 -> build_kdtree(false);
	pc1 -> build_kdtree(false);


	std::vector<PointPair> point_pairs;

	IterativeClosestPointToPlane::compute_pairs(
	point_pairs,
	pc0,
	pc1, 
	0);

	IterativeClosestPointToPlane icp(pc1, pc0);
	arma::vec y = icp.compute_y_vector(point_pairs,arma::eye<arma::mat>(3,3),arma::zeros<arma::vec>(3));

	y.save("y.txt",arma::raw_ascii);



	return 0;
}