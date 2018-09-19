#ifndef HEADER_POINTDESCRIPTOR
#define HEADER_POINTDESCRIPTOR

#include <armadillo>

class PointNormal;

class PointDescriptor{

public:

	PointDescriptor();

	arma::vec get_histogram() const;
	unsigned int get_histogram_size() const;
	double get_histogram_value(int bin_index) const;
	double distance_to(const PointDescriptor & descriptor) const;
	double distance_to(PointDescriptor * descriptor) const;

	std::vector<std::shared_ptr<PointNormal> > * get_exclusive_neighbors();

	static void compute_darboux_frames_local_hist( int & alpha_bin_index,int & phi_bin_index,int & theta_bin_index, const int & N_bins,
	const arma::vec::fixed<3> & p_i,const arma::vec::fixed<3> & n_i,const arma::vec::fixed<3> & p_j,const arma::vec::fixed<3> & n_j);

	int get_type() const;


	

protected:
	
	arma::vec histogram;
	std::vector<std::shared_ptr<PointNormal> > neighbors_exclusive;

	int type;

};





#endif