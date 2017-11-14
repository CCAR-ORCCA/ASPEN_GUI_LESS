#ifndef HEADER_SHAPEMODELBEZIER
#define HEADER_SHAPEMODELBEZIER


#include "ShapeModel.hpp"
#include "ShapeModelTri.hpp"
#include "Bezier.hpp"



class ShapeModelBezier : public ShapeModel{


public:

	/**
	Constructor
	@param shape_model pointer to polyhedral shape model used to construct 
	this new shape model
	@param frame_graph Pointer to the graph storing
	reference frame relationships
	@param frame_graph Pointer to the reference frame graph
	*/
	ShapeModelBezier(
		ShapeModelTri * shape_model,
		std::string ref_frame_name,
		FrameGraph * frame_graph);

protected:

	std::vector<std::shared_ptr<Element  > > elements;
	std::vector<std::shared_ptr< ControlPoint> >  control_points;


};













#endif