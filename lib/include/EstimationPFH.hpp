#ifndef HEADER_ESTIMATIONPFH
#define HEADER_ESTIMATIONPFH

#include <EstimationFeature.hpp>
#include <PFH.hpp>

template <class T,class U> class EstimationPFH : public EstimationFeature<T,U>{

public:

	EstimationPFH(const T & pc);
	virtual void estimate(double radius_neighbors,int N_bins,U & output_pc);


protected:

};


#endif