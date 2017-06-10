#include "Lidar.hpp"
#include "ShapeModel.hpp"
#include "ShapeModelImporter.hpp"
#include "FrameGraph.hpp"
#include "Filter.hpp"

#include <chrono>

int main() {

	// Ref frame graph
	FrameGraph frame_graph;
	frame_graph.add_frame("N");
	frame_graph.add_frame("L");
	frame_graph.add_frame("T");
	frame_graph.add_frame("E");
	frame_graph.add_transform("N", "L");
	frame_graph.add_transform("N", "T");
	frame_graph.add_transform("N", "E");


	// Shape model
	ShapeModel true_shape_model("T", &frame_graph);
	ShapeModel estimated_shape_model("E", &frame_graph);

	// ShapeModelImporter shape_io_truth(
	//     "../resources/shape_models/itokawa_8.obj",
	//     3000,false);

	// ShapeModelImporter shape_io_estimated(
	//     "../resources/shape_models/faceted_sphere.obj",
	//     200,false);


	ShapeModelImporter shape_io_truth(
	    "../resources/shape_models/KW4Alpha.obj",
	    1000);

	ShapeModelImporter shape_io_estimated(
	    "../resources/shape_models/faceted_sphere.obj",
	    400, false);

	// ShapeModelImporter shape_io_truth(
	//     "../resources/shape_models/67P_lowlowres.obj",
	//     1000,
	//     false);

	// ShapeModelImporter shape_io_estimated(
	//     "../resources/shape_models/faceted_sphere.obj",
	//     300, false);


	shape_io_truth.load_shape_model(&true_shape_model);
	shape_io_estimated.load_shape_model(&estimated_shape_model);

	// Lidar
	// arma::vec lidar_pos = { 3, 0., 0.};

	arma::vec lidar_pos = { 2000, 0., 0.};
	frame_graph.set_transform_origin("N", "L", lidar_pos);
	Lidar lidar(&frame_graph, "L", 35, 35, 32, 32, 1e-2, 1);

	// Instrument orbit (rate and inclination)
	double orbit_rate =  2 * arma::datum::pi * 1e-2 ;
	double inclination = 80 * arma::datum::pi / 180;

	// Target angular rate
	double body_spin_rate =  2e-1 ;

	// Minimum impact angle for a measurement to be used (deg)
	double min_normal_observation_angle = 10 * arma::datum::pi / 180.;

	// Minimum angular difference between two neighboring surface
	// normals to be considered different
	double min_facet_normal_angle_difference = 45 * arma::datum::pi / 180.;

	// Minimum number of rays required for a facet to be considered
	// in the estimation process
	unsigned int minimum_ray_per_facet = 5;

	// Maximum number of times a facet and its children can be split
	unsigned int max_split_count = 5;

	// Ridge estimation
	double ridge_coef = 1e1;

	// Remove outliers
	bool reject_outliers = true;

	// Activate surface splitting
	bool split_status = true;

	// Use cholesky decomposition (WARNING: SHOULD TURN OFF RIDGE IF TRUE)
	bool use_cholesky = false;

	// Recycle degenerate facets
	bool recycle_shrunk_facets = true;

	// Minimum facet angle indicating degeneracy
	double min_facet_angle = 10 * arma::datum::pi / 180;

	// Minimum edge angle indicating degeneracy
	// 0 deg: no edge recycling at all
	// 90 deg : facets cannot have their normals in opposite directions
	double min_edge_angle = 20 * arma::datum::pi / 180;

	// Time spans
	double t0 = 0;
	double tf = 500;

	// Filter arguments
	Arguments args = Arguments( t0,
	                            tf,
	                            min_normal_observation_angle,
	                            orbit_rate,
	                            inclination,
	                            body_spin_rate,
	                            min_facet_normal_angle_difference,
	                            ridge_coef,
	                            min_facet_angle,
	                            min_edge_angle,
	                            minimum_ray_per_facet,
	                            max_split_count,
	                            reject_outliers,
	                            split_status,
	                            use_cholesky,
	                            recycle_shrunk_facets);



	// Filter
	Filter filter(&frame_graph,
	              &lidar,
	              &true_shape_model,
	              &estimated_shape_model,
	              &args);

	filter.run(5, true, true);
	// filter.get_surface_point_cloud();





	return 0;
}












