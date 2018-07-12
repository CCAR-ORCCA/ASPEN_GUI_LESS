#ifndef HEADER_SHAPEMODELBEZIER
#define HEADER_SHAPEMODELBEZIER

#include "ShapeModel.hpp"
#include "Bezier.hpp"



#pragma omp declare reduction (+ : arma::mat::fixed<3,3> : omp_out += omp_in)\
initializer( omp_priv = omp_orig )

class ShapeModelTri;


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
	ShapeModelBezier(ShapeModelTri * shape_model,
		std::string ref_frame_name,
		FrameGraph * frame_graph);

	/**
	Constructor
	@param shape_model pointer to polyhedral shape model used to construct 
	this new shape model
	@param frame_graph Pointer to the graph storing
	reference frame relationships
	@param frame_graph Pointer to the reference frame graph
	@param surface_noise use in patch covariance setting
	*/
	ShapeModelBezier(ShapeModelTri * shape_model,
	std::string ref_frame_name,
	FrameGraph * frame_graph,double surface_noise);

	/**
	Constructor
	@param frame_graph Pointer to the graph storing
	reference frame relationships
	@param frame_graph Pointer to the reference frame graph
	*/
	ShapeModelBezier(std::string ref_frame_name,
		FrameGraph * frame_graph);

	/**
	Constructor.
	Takes a single patch as argument
	@param patch pointer to patch the shape model will be comprised off
	*/
	ShapeModelBezier(Bezier patch);

	std::shared_ptr<arma::mat> get_info_mat_ptr() const;
	std::shared_ptr<arma::vec> get_dX_bar_ptr() const;

	/**
	Constructs a kd tree for ray-tracing purposes
	*/
	virtual void construct_kd_tree_shape();


	/**
	Updates the values of the center of mass, volume
	*/
	virtual void update_mass_properties();


	void initialize_info_mat();
	void initialize_dX_bar();

	arma::mat random_sampling(unsigned int N,const arma::mat & R = 1e-6 * arma::eye<arma::mat>(3,3)) const;


	/**
	Saves the shape model to an obj file as a polyhedron
	*/
	void save_to_obj(std::string path) ;


	/**
	Saves the shape model as a collection of Bezier patches
	*/
	void save(std::string path) ;

	/**
	Elevates the degree of all Bezier patches in the shape model by one
	@param update true if the mass properties/kd tree of the shape model should be updated , false otherwise
	*/
	void elevate_degree(bool update = true);

	/**
	Gets the shape model degree
	*/
	unsigned int get_degree();

	/**
	Computes the surface area of the shape model
	*/
	virtual void compute_surface_area();
	/**
	Computes the volume of the shape model
	*/
	virtual void compute_volume();
	/**
	Computes the center of mass of the shape model
	*/
	virtual void compute_center_of_mass();
	/**
	Computes the inertia tensor of the shape model
	*/
	virtual void compute_inertia();

	/**
	Finds the intersect between the provided ray and the shape model
	@param ray pointer to ray. If a hit is found, the ray's internal is changed to store the range to the hit point
	*/
	virtual bool ray_trace(Ray * ray);

	void save_both(std::string partial_path);


	/**
	Assembles the tables used to compute the mass properties of the shape
	*/
	void populate_mass_properties_coefs();


	/**
	Compute the standard deviation in the volume
	*/
	void compute_volume_sd();

	/**
	Compute the covariance in the center of mass
	*/
	void compute_cm_cov();

	/**
	Runs a Monte Carlo on volume
	@param N number of runs
	@return results
	*/
	arma::vec run_monte_carlo_volume(int N);


	/**
	Runs a Monte Carlo on volume
	@param N number of runs
	@return results
	*/
	arma::mat run_monte_carlo_cm(int N);





	/**
	Return the standard deviation in the volume
	@return standard deviation in shape volume
	*/
	double get_volume_sd() const;


	/**
	Return the center of mass covariance 
	@return center of mass covariance 
	*/
	arma::mat get_cm_cov() const;







	/**
	Generates all the possible combinations of indices as needed in the inertia moments
	computation from the tensorization of the base vector
	@param n_indices number of indices to tensorize
	@parma base_vector vector of indices to tensorize (contains N such indices pairs)
	@param index_vectors container holding the final N ^ n_indices vectors of indices pairs
	@param temp_vector container holding the current vector of indices being built

	@param depth depth of the current vector
	*/
	static void build_bezier_index_vectors(const int & n_indices,
		const std::vector<std::vector<int> > & base_vector,
		std::vector<std::vector<std::vector<int> > > & index_vectors,
		std::vector < std::vector<int> > & temp_vector,
		const int depth = 0);


	/**	
	Builds the vector holding the pair of indices of each point in a Bezier triangle
	@param n Bezier triangle degree
	@param base_vector container to hold the pair of indices of each point in a Bezier triangle
	*/
	static void build_bezier_base_index_vector(const int n,std::vector<std::vector<int> > & base_vector);


protected:



	arma::mat::fixed<3,3> increment_cm_cov(arma::mat::fixed<12,3> & left_mat,
		arma::mat::fixed<12,3>  & right_mat, 
		int i,int j,int k,int l, 
		int m, int p, int q, int r);

	void construct_mat(arma::mat::fixed<12,3> & mat,
		int i,int j,int k,int l);

	std::shared_ptr<arma::mat> info_mat_ptr = nullptr;
	std::shared_ptr<arma::vec> dX_bar_ptr = nullptr;
	std::vector<std::vector<double> > cm_gamma_indices_coefs_table;
	std::vector<std::vector<double> > volume_indices_coefs_table;
	std::vector<std::vector<double> > inertia_indices_coefs_table;

	std::vector<std::vector<double> > volume_sd_indices_coefs_table;
	std::vector<std::vector<double> > cm_cov_1_indices_coefs_table;
	std::vector<std::vector<double> > cm_cov_2_indices_coefs_table;



	double volume_sd;
	arma::mat cm_cov = arma::zeros<arma::mat>(3,3);



	
	
};













#endif