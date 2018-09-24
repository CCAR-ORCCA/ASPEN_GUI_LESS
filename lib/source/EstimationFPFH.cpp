#include <EstimationFPFH.hpp>
#include <PointNormal.hpp>
#include <PointDescriptor.hpp>



template <class T,class U>
EstimationFPFH<T,U>::EstimationFPFH(const T & pc) : EstimationFeature<T,U>(pc){

}



template<class T,class U>
void EstimationFPFH<T,U>::estimate(double radius_neighbors,int N_bins,U & output_pc){

	if (radius_neighbors < 0){
		throw(std::runtime_error("neighborhood_radius is negative"));
	}

	unsigned int size = this -> pc. size();
	assert(size == output_pc.size());

	std::vector<std::vector<int> > all_points_neighborhoods;
	std::vector<SPFH> all_point_spfhs;
	all_points_neighborhoods.resize(size);
	all_point_spfhs.resize(size);

	#pragma omp parallel for
	for (unsigned int i = 0; i < size; ++i) {
		std::vector<int> neighborhood = this -> pc.get_nearest_neighbors_radius(this -> pc.get_point_coordinates(i),radius_neighbors);
		all_points_neighborhoods[i] = neighborhood;
		all_point_spfhs[i] = SPFH(i,neighborhood,N_bins,this -> pc);
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < size; ++i) {
		output_pc[i] = FPFH(i,
			all_points_neighborhoods,
			all_point_spfhs,
			this -> pc,
			this -> scale_distance);
	}

}

template<class T,class U>
void EstimationFPFH<T,U>::set_scale_distance(bool scale_distance){
	this -> scale_distance = scale_distance;
}

template class EstimationFPFH<PointCloud<PointNormal>,PointCloud<PointDescriptor> >;
