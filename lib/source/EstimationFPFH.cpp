#include <EstimationFPFH.hpp>
#include <PointNormal.hpp>
#include <PointDescriptor.hpp>



template <class T,class U>
EstimationFPFH<T,U>::EstimationFPFH(const PointCloud<T> & input,PointCloud<U> & output_pc) : EstimationFeature<T,U>(input,output_pc){

}

template<class T,class U>
void EstimationFPFH<T,U>::estimate(double radius_neighbors,bool force_use_previous){

	if (radius_neighbors < 0){
		throw(std::runtime_error("neighborhood_radius is negative"));
	}

	unsigned int size = this -> input_pc. size();
	assert(size == this -> output_pc.size());

	std::vector<std::vector<int> > all_points_neighborhoods;
	std::vector<SPFH> all_point_spfhs;
	all_points_neighborhoods.resize(size);
	all_point_spfhs.resize(size);

	#pragma omp parallel for
	for (unsigned int i = 0; i < size; ++i) {
		std::vector<int> neighborhood = this -> input_pc.get_nearest_neighbors_radius(this -> input_pc.get_point_coordinates(i),radius_neighbors);
		all_points_neighborhoods[i] = neighborhood;
		all_point_spfhs[i] = SPFH(i,neighborhood,this -> N_bins,this -> input_pc);
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < size; ++i) {
		this -> output_pc[i] = FPFH(i,
			all_points_neighborhoods,
			all_point_spfhs,
			this -> input_pc,
			this -> scale_distance,
			this -> output_pc[i].get_is_valid_feature());
	}

}




template<class T,class U>
void EstimationFPFH<T,U>::set_scale_distance(bool scale_distance){
	this -> scale_distance = scale_distance;
}


template<class T,class U>
void EstimationFPFH<T,U>::set_N_bins(int N_bins){
	this -> N_bins = N_bins;
}


template<class T,class U>
void EstimationFPFH<T,U>::estimate(int N_neighbors,bool force_use_previous){
	throw(std::runtime_error("EstimationFPFH<T,U>::estimate(int N_neighbors) is not implemented"));
}

template class EstimationFPFH<PointNormal,PointDescriptor> ;
