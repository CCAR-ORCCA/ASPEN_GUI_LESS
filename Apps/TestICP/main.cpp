
#include "ICPBase.hpp"
#include <armadillo>
#include "boost/progress.hpp"
#include <ShapeModelImporter.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>
#include <IterativeClosestPointToPlane.hpp>
#include <IterativeClosestPointToPlane.hpp>
#include <PointNormal.hpp>
#include <EstimationNormals.hpp>
#include <PointCloudIO.hpp>

int main() {

	std::shared_ptr<PointCloud<PointNormal >> destination_pc = std::make_shared<PointCloud<PointNormal >> (PointCloud<PointNormal >("../pc/destination.txt",true));
	std::shared_ptr<PointCloud<PointNormal >> source_pc = std::make_shared<PointCloud<PointNormal >> (PointCloud<PointNormal >("../pc/source.txt",true));
	destination_pc -> build_kdtree(false);
	source_pc -> build_kdtree(false);


	arma::vec::fixed<3> los = {1,0,0};
	EstimationNormals<PointNormal,PointNormal> estimate_normals(*destination_pc,*destination_pc);
	estimate_normals.set_los_dir(los);
	estimate_normals.estimate(6);

	EstimationNormals<PointNormal,PointNormal> estimate_normals_2(*source_pc,*source_pc);
	estimate_normals_2.set_los_dir(los);
	estimate_normals_2.estimate(6);

	IterativeClosestPointToPlane icp_pc(destination_pc,source_pc);
	icp_pc.register_pc(1e-2,arma::eye<arma::mat>(3,3));


	PointCloudIO<PointNormal>::save_to_obj(*destination_pc, "../output/destination.obj");
	PointCloudIO<PointNormal>::save_to_obj(*source_pc, "../output/source_pc.obj");
	PointCloudIO<PointNormal>::save_to_obj(*source_pc, "../output/registered_source_pc.obj",icp_pc.get_dcm(),icp_pc.get_x());

	return 0;
}












