#ifndef HEADER_LIDAR
#define HEADER_LIDAR

#include "ShapeModel.hpp"
#include "Ray.hpp"
#include "FrameGraph.hpp"
#include "GNUPlot.h"

class Ray;
class ShapeModel;

class Lidar {

public:

	/**
	Constructor. Allocates memory to each of the rays cast by
	the instrument
	@param frame_graph Pointer to the reference frame graph
	@param fov_h horizontal field of view (degrees)
	@param fov_v vertical field of view (degrees)
	@param res_y horizontal resolution (number of pixel rows)
	@param res_z vertical resolution (number of pixel columns)
	@param f focal length (m)
	@param freq frequency of operation (Hz)
	*/
	Lidar(FrameGraph * frame_graph,
	      std::string ref_frame_name = "L",
	      double fov_h = 10,
	      double fov_v = 10,
	      unsigned int res_y = 16,
	      unsigned int res_z = 16,
	      double f = 1e-2,
	      double freq = 3
	     );



	/**
	Get the instrument focal length
	@return focal length (m)
	*/
	double get_focal_length() const;

	/**
	Get the instrument horizontal fov
	@param use_rad
	@return horizontal fov (degrees or radians)
	*/
	double get_fov_y(bool use_rad = true) const;

	/**
	Get the instrument vertical fov
	@param use_rad
	@return vertical fov (degrees or radians)
	*/
	double get_fov_z(bool use_rad = true) const;

	/**
	Get the vertical focal plane size
	@return focal plane vertical size (m)
	*/
	double get_size_z() const ;

	/**
	Get the horizontal focal plane size
	@return focal plane horizontal size (m)
	*/
	double get_size_y() const ;

	/**
	Get the resolution of the focal plane along the y axis
	@return resolution along the y axis
	*/
	double get_res_y() const ;

	/**
	Get the resolution of the focal plane along the z axis
	@return resolution along the z axis
	*/
	double get_res_z() const ;

	/**
	Returns the frequency of operation
	@return Frequency (Hz)
	*/
	double get_frequency() const;


	/**
	Writes the location of the pixels to a file
	@param path Path to the file to store the pixel locations
	*/
	void save_pixel_location(std::string path) const ;


	/**
	Plot the ranges collected in the the focal plane
	after storing them at the provided location
	@param path Path to the text file containing the ranges
	*/
	void plot_ranges(std::string path) const;


	/**
	Sends a laser flash to the targeted shape model.
	Every pixel present in the focal plane will cast a ray towards the target.
	Depending on whether the target is hit or not,
	the corresponding members of the cast ray are updated
	@param shape_model Pointer to the shape model being observed
	*/
	void send_flash(ShapeModel * shape_model);

	/**
	Accessor to the shape model currently observed
	@return pointer to shape model currently observed
	*/
	ShapeModel * get_shape_model();



	/*
	Saves the ranges collected by each pixel in the focal plane
	to a file
	@param path Path to the file
	@return Pair of min and max measured range
	*/
	std::pair<double, double> save_focal_plane_range(std::string path) const ;


	FrameGraph * get_frame_graph();

	std::string get_ref_frame_name() const;

protected:
	double f;
	double freq;

	double fov_y ;
	double fov_z ;
	double res_y ;
	double res_z ;

	FrameGraph * frame_graph;
	std::string ref_frame_name;

	ShapeModel * shape_model = nullptr;
	std::vector<std::vector<std::shared_ptr<Ray> > > focal_plane;


};


#endif