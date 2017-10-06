#ifndef HEADER_SHAPEMODELBEZIER
#define HEADER_SHAPEMODELBEZIER


#include "ShapeModel.hpp"

class ShapeModelBezier : public ShapeModel{


public:

	/**
	Constructor
	@param frame_graph Pointer to the graph storing
	reference frame relationships
	@param frame_graph Pointer to the reference frame graph
	*/
	ShapeModelBezier(): ShapeModel(ref_frame_name,frame_graph){};

protected:

	std::vector<std::shared_ptr<Element  > > elements;
	std::vector<std::shared_ptr< ControlPoint> >  control_points;


};













#endif