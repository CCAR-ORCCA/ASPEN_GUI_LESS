#include <EstimationFeature.hpp>
#include <PointNormal.hpp>
#include <PointDescriptor.hpp>


template <class T,class U>
EstimationFeature<T,U>::EstimationFeature(const T & pc) : pc(pc){

}

template class EstimationFeature<PointCloud<PointNormal>,PointCloud<PointDescriptor> >;
