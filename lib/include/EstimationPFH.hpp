#ifndef HEADER_ESTIMATIONPFH
#define HEADER_ESTIMATIONPFH

#include <EstimationFeature.hpp>
#include <PFH.hpp>

template <class T,class U> class EstimationPFH : public EstimationFeature<T,U>{

public:

	EstimationPFH(const PointCloud<T> & input_pc,PointCloud<U> & output_pc);
	virtual void estimate(double radius_neighbors);
	virtual	void estimate(int N_neighbors);

	void set_N_bins(int N_bins);


protected:
	int N_bins = 11;

};


#endif