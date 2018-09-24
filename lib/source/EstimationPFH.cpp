#include <EstimationPFH.hpp>
#include <PointNormal.hpp>

template <class T,class U>
EstimationPFH<T,U>::EstimationPFH(const T & pc) : EstimationFeature<T,U>(pc){

}


template<class T,class U>
void EstimationPFH<T,U>::estimate(double radius_neighbors,int N_bins,U & output_pc){

	unsigned int size = this -> pc. size();
	assert(size == output_pc.size());
	if (radius_neighbors < 0){
		throw(std::runtime_error("neighborhood_radius is negative"));
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < size; ++i) {
		std::vector<int> neighborhood = this -> pc.get_nearest_neighbors_radius(this -> pc.get_point_coordinates(i),radius_neighbors);
		output_pc[i] = PFH(neighborhood,N_bins,this -> pc);
	}	

}



template class EstimationPFH<PointCloud<PointNormal>,PointCloud<PointDescriptor> >;
