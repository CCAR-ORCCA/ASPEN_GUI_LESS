#include <CGAL_interface.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <PointCloudIO.hpp>


int main(){
	
	std::string pc_original_path = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/159_coverage_pc.obj";
	std::string pc_cgal_path = "../output/pc_cgal.txt";
	std::string shape_cgal_path = "../output/shape_cgal.obj";


	double percentage_point_kept = 1;
	Kernel::FT sm_angle = 30.0;
	Kernel::FT sm_radius = 3;
	Kernel::FT sm_distance = 3;

	PointCloud<PointNormal> pc(pc_original_path);

	PointCloud<PointNormal> pc_downsampled;

	std::vector<int> indices;
	for (int i = 0; i < pc.size(); ++i){
		indices.push_back(i);
	}

	std::cout << "Shuffling\n";
	std::random_shuffle ( indices.begin(), indices.end() );

	int points_kept = static_cast<int>(percentage_point_kept/100. * pc.size() );

	for (int i = 0; i < points_kept; ++i){
		pc_downsampled.push_back(pc[indices[i]]);
	}

	std::cout << "Saving to txt\n";

	PointCloudIO<PointNormal>::save_to_txt(pc_downsampled, pc_cgal_path);

	// sm_angle = 30.0; // Min triangle angle in degrees.
	// sm_radius = 30; // Max triangle size w.r.t. point set average spacing.
	// sm_distance = 0.5; // Surface Approximation error w.r.t. point set average spacing.

	
	CGALINTERFACE::CGAL_interface(pc_cgal_path.c_str(), shape_cgal_path.c_str(),4000,
		sm_angle,
		sm_radius,
		sm_distance);
	
	return 1;
}