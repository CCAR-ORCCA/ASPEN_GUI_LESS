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

	FrameGraph frame_graph;

	ShapeModelTri<ControlPoint> psr_shape("", &frame_graph);

	PointCloud<PointNormal> global_pc("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/119_coverage_pc_as_is.obj");
	ShapeModelImporter::load_obj_shape_model("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0/shape_cgal.obj",1.,true,psr_shape);
	psr_shape.construct_kd_tree_shape();

	ShapeModelBezier<ControlPoint> bezier_shape(psr_shape,"", &frame_graph);
	bezier_shape.elevate_degree();

	ShapeFitterBezier shape_fitter(&psr_shape,&bezier_shape,&global_pc); 
	shape_fitter.fit_shape_batch(3,0);
	bezier_shape.save_both("../fit_shape");

	return 0;
}












