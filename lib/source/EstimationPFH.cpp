#include <EstimationPFH.hpp>
#include <PointNormal.hpp>

template <class T,class U>
EstimationPFH<T,U>::EstimationPFH(const PointCloud<T> & input_pc,PointCloud<U> & output_pc) : EstimationFeature<T,U>(input_pc,output_pc){

}

template<class T,class U>
void EstimationPFH<T,U>::estimate(double radius_neighbors,bool force_use_previous){

	unsigned int size = this -> input_pc. size();
	assert(size == this -> output_pc.size());
	if (radius_neighbors < 0){
		throw(std::runtime_error("neighborhood_radius is negative"));
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < size; ++i) {
		std::vector<int> neighborhood = this -> input_pc.get_nearest_neighbors_radius(this -> input_pc.get_point_coordinates(i),radius_neighbors);
		this -> output_pc[i] = PFH(neighborhood,this -> N_bins,this -> input_pc);
	}	
}

template<class T,class U>
void EstimationPFH<T,U>::set_N_bins(int N_bins){
	this -> N_bins = N_bins;
}


template<class T,class U>
void EstimationPFH<T,U>::estimate(int N_neighbors,bool force_use_previous){
	throw(std::runtime_error("EstimationPFH<T,U>::estimate(int N_neighbors) is not implemented"));
}


template class EstimationPFH<PointNormal,PointDescriptor> ;
