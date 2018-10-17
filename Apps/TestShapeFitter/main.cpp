#include <Lidar.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelImporter.hpp>
#include <ShapeBuilder.hpp>
#include <Observations.hpp>
#include <Dynamics.hpp>
#include <Observer.hpp>
#include <System.hpp>
#include <ShapeBuilderArguments.hpp>
#include <StatePropagator.hpp>
#include <ShapeFitterBezier.hpp>


#include <NavigationFilter.hpp>
#include <SBGATSphericalHarmo.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>


#include <chrono>
#include <boost/progress.hpp>
#include <boost/numeric/odeint.hpp>
#include <vtkOBJReader.h>



int main() {

	
	// ShapeModelBezier<ControlPoint> bezier_shape;

	FrameGraph frame_graph;

	ShapeModelTri<ControlPoint> psr_shape("", &frame_graph);
	ShapeModelBezier<ControlPoint> bezier_shape("", &frame_graph);

	PointCloud<PointNormal> global_pc("../global_pc.obj");
	ShapeModelImporter::load_obj_shape_model("../shape_cgal.obj", 
		1, true,psr_shape);
	psr_shape.construct_kd_tree_shape();

	ShapeFitterBezier shape_fitter(&psr_shape,&bezier_shape,&global_pc); 
	shape_fitter.fit_shape_batch(10,0);

	return 0;
}












