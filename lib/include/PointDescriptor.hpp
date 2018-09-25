#ifndef HEADER_POINTDESCRIPTOR
#define HEADER_POINTDESCRIPTOR

#include <armadillo>
#include <memory>

class PointNormal;


class PointDescriptor{

public:

	PointDescriptor();
	PointDescriptor(arma::vec histogram);




	const arma::vec & get_histogram() const;

	unsigned int get_histogram_size() const;
	double get_histogram_value(int bin_index) const;

	double distance_to_descriptor(const PointDescriptor & descriptor) const;
	double distance_to_descriptor(const arma::vec & histogram) const;

	static void compute_darboux_frames_local_hist( int & alpha_bin_index,int & phi_bin_index,int & theta_bin_index, const int & N_bins,
		const arma::vec::fixed<3> & p_i,const arma::vec::fixed<3> & n_i,const arma::vec::fixed<3> & p_j,const arma::vec::fixed<3> & n_j);

	int get_type() const;
	enum Type{PFHDescriptor,FPFHDescriptor};

	int get_global_index() const;
	void set_global_index(int index);

	bool get_is_valid_feature() const;
	void set_is_valid_feature(bool valid_feature);

	

protected:
	
	arma::vec histogram;

	int type;
	int global_index;
	bool is_valid_feature = true;


};





#endif