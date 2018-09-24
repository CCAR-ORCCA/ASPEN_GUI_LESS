#include <EstimationNormals.hpp>
#include <PointNormal.hpp>

template <class T>
EstimationNormals<T>::EstimationNormals(T & pc) : pc(pc){
	this -> pc = pc;
}


template <>
void EstimationNormals<PointCloud<PointNormal> >::estimate_normals(int N_neighbors,const arma::vec::fixed<3> & los_dir){
	
	#pragma omp parallel for 
	for (unsigned int i = 0; i < this -> pc .  size(); ++i) {

		// Get the N nearest neighbors to this point
		auto closest_points = this -> pc.get_closest_N_points(this -> pc.get_point_coordinates(i), N_neighbors);

		arma::mat::fixed<3,3> covariance = arma::zeros<arma::mat>(3,3);
		arma::vec::fixed<3> centroid = {0,0,0};

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			centroid += this -> pc.get_point_coordinates(it -> second)/closest_points.size();
		}

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			const arma::vec & p = this -> pc.get_point_coordinates(it -> second);
			covariance += 1./(closest_points.size() - 1) * (p - centroid) * (p - centroid).t();
		}

		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, covariance);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		if (arma::dot(n, los_dir) < 0) {
			this -> pc.get_point(i).set_normal_coordinates(n);
		}
		else {
			this -> pc.get_point(i).set_normal_coordinates(-n);
		}
	}
	
}



template <>
void EstimationNormals<PointCloud<PointNormal> >::estimate_normals(double radius,const arma::vec::fixed<3> & los_dir){
	
	#pragma omp parallel for 
	for (unsigned int i = 0; i < this -> pc .  size(); ++i) {

		// Get the N nearest neighbors to this point

		auto closest_points = this -> pc.get_nearest_neighbors_radius(this -> pc.get_point_coordinates(i), radius);

		arma::mat::fixed<3,3> covariance = arma::zeros<arma::mat>(3,3);
		arma::vec::fixed<3> centroid = {0,0,0};

		int size = closest_points.size();
		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			centroid += this -> pc.get_point_coordinates(*it)/size;
		}

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			const arma::vec & p = this -> pc.get_point_coordinates(*it);
			covariance += 1./(size - 1) * (p - centroid) * (p - centroid).t();
		}

		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, covariance);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		if (arma::dot(n, los_dir) < 0) {
			this -> pc.get_point(i).set_normal_coordinates(n);
		}
		else {
			this -> pc.get_point(i).set_normal_coordinates(-n);
		}
	}
	
}








template class EstimationNormals<PointCloud<PointNormal>>;
