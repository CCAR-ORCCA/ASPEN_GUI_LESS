#include "Lidar.hpp"
#include "ShapeModel.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"

int main() {

	// Shape model
	ShapeModel shape_model;
	ShapeModelImporter shape_io("../resources/shape_models/saturn_v_ft.obj");
	shape_io.load_shape_model(&shape_model);

	// Lidar
	Lidar lidar(10, 10, 2, 1, 1e-2);
	lidar.set_shape_model(&shape_model);

	// Ref frame graph
	FrameGraph frame_graph;


	// Test
	// Add frame, add transform, set transform origin and orientation


	frame_graph.add_frame("ECI");
	frame_graph.add_frame("ECF");
	// frame_graph.add_frame("E");


	frame_graph.add_transform("ECI", "ECF");
	// frame_graph.add_transform("ECI", "E");
	// frame_graph.add_transform("E", "ECF");


	arma::vec angles = {2.3508, 48.8567};
	angles = arma::datum::pi * angles / 180;

	arma::vec angles_1 = {2.3508, 0};
	arma::vec angles_2 = {0, 48.8567};

	angles_1 = arma::datum::pi * angles_1 / 180;
	angles_2 = arma::datum::pi * angles_2 / 180;

	arma::vec mrp_FI = RBK::dcm_to_mrp(longitude_latitude_to_dcm(angles));

	arma::vec mrp_EI = RBK::dcm_to_mrp(longitude_latitude_to_dcm(angles_1));
	arma::vec mrp_FE = RBK::dcm_to_mrp(longitude_latitude_to_dcm(angles_2));

	arma::vec origin_LB = {0, 0, 0};

	arma::vec p_ECF = {6378, 0, 0};
	arma::vec p_ECI = {4.1928e+03,   1.7213e+02 ,  4.8031e+03};

	frame_graph.set_transform_mrp("ECI", "ECF", mrp_FI);
	frame_graph.set_transform_origin("ECI", "ECF", origin_LB);

	// frame_graph.set_transform_mrp("ECI", "E", mrp_EI);
	// frame_graph.set_transform_origin("ECI", "E", origin_LB);

	// frame_graph.set_transform_mrp("E", "ECF", mrp_FE);
	// frame_graph.set_transform_origin("E", "ECF", origin_LB);

	// std::cout << frame_graph.convert(p_ECF, "ECF", "ECI").t() << std::endl;
	std::cout << frame_graph.convert(p_ECI, "ECI", "ECF").t() << std::endl;

	// arma::vec temp =  frame_graph.convert(p, "ECF", "E");
	// std::cout << frame_graph.convert(temp, "E", "ECI").t() << std::endl;


	return 0;
}












