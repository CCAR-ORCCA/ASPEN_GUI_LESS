#include <EstimationFeature.hpp>
#include <PointNormal.hpp>
#include <PointDescriptor.hpp>


template <class T,class U>
EstimationFeature<T,U>::EstimationFeature(const PointCloud<T> & input_pc,PointCloud<U> & output_pc) : input_pc(input_pc), output_pc(output_pc){

}

template<class T,class U>
void EstimationFeature<T,U>::compute_center(){
	this -> center = EstimationFeature<T,U>::compute_center(this -> output_pc);
}

template<class T,class U>
arma::vec EstimationFeature<T,U>::compute_center(const PointCloud<U> & pc){
	arma::vec center = arma::zeros<arma::vec>(pc.get_point_coordinates(0).size());

	// #pragma omp parallel for
	for (unsigned int i = 0; i < pc.size(); ++i) {
		center += pc.get_point_coordinates(i);
	}

	center *= 1./pc.size();
	return center;
}

template<class T,class U>
arma::mat EstimationFeature<T,U>::compute_principal_axes(const PointCloud<U> & pc,const arma::vec & center){
	
	arma::mat cov = arma::zeros<arma::vec>(pc.get_point_coordinates(0).size(),
		pc.get_point_coordinates(0).size());

	// #pragma omp parallel for
	for (unsigned int i = 0; i < pc.size(); ++i) {
		cov += (pc.get_point_coordinates(i) - center) * (pc.get_point_coordinates(i) - center).t();
	}

	cov *= 1./(pc.size() - 1);
	arma::vec eigval;
	arma::mat eigvec;

	if(!arma::eig_sym( eigval, eigvec, cov )){
		throw(std::runtime_error("Principal axes computation failed in EstimationFeature<T,U>::compute_principal_axes"));
	}

	// Enforcing determinant of eigvec to be positive
	if (arma::det(eigvec) < 0){
		eigvec.col(0) *= -1;
	}

	return eigvec;
}



template<>
arma::vec EstimationFeature<PointNormal,PointDescriptor>::compute_distances_to_center(const arma::vec & center,
	const PointCloud<PointDescriptor> & pc){

	arma::vec distances_to_center(pc.size());
	PointDescriptor mean_descriptor(center);

	#pragma omp parallel for
	for (int i = 0 ; i < pc.size(); ++i){
		distances_to_center(i) = mean_descriptor.distance_to_descriptor(pc.get_point(i));
	}

	distances_to_center = arma::abs(distances_to_center - arma::mean(distances_to_center))/arma::stddev(distances_to_center);

	return distances_to_center;
}

template<>
arma::vec EstimationFeature<PointNormal,PointDescriptor>::compute_distances_to_center(){
	return EstimationFeature<PointNormal,PointDescriptor>::compute_distances_to_center(this -> center,this -> output_pc);
}


template<>
void EstimationFeature<PointNormal,PointDescriptor>::disable_common_features(double const & beta, 
	const arma::vec & distances,
	PointCloud<PointDescriptor> & pc){
	
	#pragma omp parallel for
	for (int i = 0 ; i < pc.size(); ++i){
		if (distances(i) < beta){
			pc.get_point(i).set_is_valid_feature(false);
		}
	}

}

template<>
void EstimationFeature<PointNormal,PointDescriptor>::disable_common_features(double const & beta, const arma::vec & distances){
	EstimationFeature<PointNormal,PointDescriptor>::disable_common_features(beta,distances,this -> output_pc);
}


template <>
void EstimationFeature<PointNormal,PointDescriptor>::prune(double deadband){

	this -> compute_center();
	auto distances = this -> compute_distances_to_center();
	this -> disable_common_features(deadband,distances);

}











template class EstimationFeature<PointNormal,PointDescriptor> ;
template class EstimationFeature<PointNormal,PointNormal> ;

