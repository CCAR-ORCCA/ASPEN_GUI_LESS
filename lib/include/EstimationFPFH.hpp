#ifndef HEADER_ESTIMATIONFPH
#define HEADER_ESTIMATIONFPH

#include <EstimationFeature.hpp>
#include <FPFH.hpp>


template <class T,class U> class EstimationFPFH : public EstimationFeature<T,U>{

public:

	EstimationFPFH(const PointCloud<T> & input_pc,PointCloud<U> & output_pc);
	virtual void estimate(double radius_neighbors,bool force_use_previous = false);
	virtual	void estimate(int N_neighbors,bool force_use_previous = false);

	/**
	Toggles normalization of spfh
	@param scale_distance If true, will use normalized weights L/w_k where L is the smallest non-zero distance between a query point and its 
	neighbors 
	*/
	void set_scale_distance(bool scale_distance);

	/**
	Sets number of bins
	@param N_bins number of bins
	*/
	void set_N_bins(int N_bins);

	



protected:
	bool scale_distance = false;
	int N_bins = 11;
	double beta = 1.5;

};


#endif