#ifndef HEADER_LIDAR
#define HEADER_LIDAR

#include "OMP_flags.hpp"
#include <assert.h>
#include <string>
#include <utility>
#include <map>
#include <vector>
#include <memory>
#include <armadillo>





class Ray;
class ShapeModel;
class FrameGraph;
class PC;
class Facet;


class Lidar {

public:

	/**
	Constructor. Allocates memory to each of the rays cast by
	the instrument
	@param frame_graph Pointer to the reference frame graph
	@param fov_h horizontal field of view (degrees)
	@param fov_v vertical field of view (degrees)
	@param y_res horizontal resolution (number of pixel columns)
	@param z_res vertical resolution (number of pixel rows)
	@param f focal length (m)
	@param freq frequency of operation (Hz)
	@param los_noise_1sd_baseline 1 standard deviation of the baseline line-of-sight gaussian noise
	Total sd is given by 3 sigma = los_noise_sd_baseline + los_noise_fraction_mes_truth * rho_truth
	@param los_noise_fraction_mes_truth truth-proportional fraction of the range measurement error
	*/
	Lidar(FrameGraph * frame_graph,
		std::string ref_frame_name ,
		double fov_h ,
		double fov_v ,
		unsigned int y_res ,
		unsigned int z_res ,
		double f ,
		double freq ,
		double los_noise_1sd_baseline ,
		double los_noise_fraction_mes_truth
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
	Get the number of pixel rows in the focal plane
	@return number of pixel rows in the focal plane
	*/
	double get_z_res() const ;

	/**
	Get the number of pixel columns in the focal plane
	@return number of pixel columns in the focal plane
	*/
	double get_y_res() const ;

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
	Returns a pointer to the ray (pixel)
	@param pixel index index of ray
	@return ray pointer to the ray
	*/
	Ray * get_ray(unsigned int pixel);


	/**
	Plot the ranges collected in the the focal plane
	after storing them at the provided location
	@param path Path to the text file containing the ranges
	@param type Indicates what type of ranges should be saved
		- 0 : true ranges
		- 1 : computed ranges
		- 2 : residuals
	*/
	void plot_ranges(std::string path, unsigned int type) const;


	
	/*
	Saves the range residuals collected by each pixel in the focal plane
	to a file
	@param path Path to the file
	@return Pair of min and max measured range
	*/
	std::pair<double, double> save_range_residuals(std::string path) const ;


	/**
	Sends a laser flash to the targeted shape model.
	Every pixel present in the focal plane will cast a ray towards the target.
	Depending on whether the target is hit or not,
	the corresponding members of the cast ray are updated
	@param shape_model Pointer to the shape model being observed
	@param skip_factor determines the number of pixels skipped between two active pixels. 
	default is skip_factor == 1 (all pixels are used)
	*/

	void send_flash(ShapeModel * shape_model,bool add_noise,double skip_factor = 1) ;


	/**
	Saves the range residuals associated with each facet to
	a file
	@param path Path to the file
	@param facets_to_residuals Map storing the facet that were seen and their associated residuals

	*/
	void save_range_residuals_per_facet(std::string path, std::map<Facet * , std::vector<double> > & facets_to_residuals) const ;


	/**
	Plots the range residuals associated with each facet
	@param path Path to the file
	*/
	void plot_range_residuals_per_facet(std::string path) ;

	/**
	Returns the number of rays that hit the target
	@return number of rays that have hit the target
	*/
	unsigned int get_number_of_hits() const;


	/**
	Accessor to the shape model currently observed
	@return pointer to shape model currently observed
	*/
	ShapeModel * get_shape_model();

	FrameGraph * get_frame_graph();

	std::string get_ref_frame_name() const;

	void save(std::string path,bool conserve_format = false)  ;

	std::vector<std::shared_ptr<Ray> > * get_focal_plane() ;




protected:
	double f;
	// double freq;

	double fov_y ;
	double fov_z ;
	double z_res ;
	double y_res ;
	double los_noise_sd_baseline;
	double los_noise_fraction_mes_truth;

	FrameGraph * frame_graph;
	std::string ref_frame_name;

	ShapeModel * shape_model = nullptr;
	std::vector<std::shared_ptr<Ray> > focal_plane;

	std::vector<arma::vec> surface_measurements;





};


#endif