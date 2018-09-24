#ifndef HEADER_ESTIMATIONFPH
#define HEADER_ESTIMATIONFPH

#include <EstimationFeature.hpp>
#include <FPFH.hpp>


template <class T,class U> class EstimationFPFH : public EstimationFeature<T,U>{

public:

	EstimationFPFH(const T & pc);
	virtual void estimate(double radius_neighbors,int N_bins,U & output_pc);

	void set_scale_distance(bool scale_distance);
protected:
	bool scale_distance = false;

};


#endif