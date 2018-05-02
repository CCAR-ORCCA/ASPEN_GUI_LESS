#ifndef FLYOVER_MAP_HEADER
#define FLYOVER_MAP_HEADER

#include <set>
#include <vector>
#include <iostream>




class FlyOverMap{
public:
	FlyOverMap(int n_bins_longitude = 72,int n_bins_latitude = 36);

	void add_label(int label, double longitude, double latitude);



	void get_flyovers_in_bin(int bin_longitude,int bin_latitude,
		std::set<std::set<int> > & flyovers) const;

	bool has_flyovers(double longitude, double latitude) const ;


	std::set<std::set< int> > get_flyovers() const;


protected:

	int n_bins_longitude;
	int n_bins_latitude;
	double d_bin_longitude;
	double d_bin_latitude;

	std::vector<int> get_bin(int bin_longitude,int bin_latitude ) const;

	int get_bin_depth(int bin_longitude,int bin_latitude ) const;


	std::vector< std::vector< std::vector< int > > > bins;


};










#endif