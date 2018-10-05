#include <BSpline.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <PointCloudIO.hpp>

#define BSPLINE_DEBUG 1


BSpline::BSpline(int l,int p){

	// Setting internals
	this -> p = p;

	this -> m = l + 2 * p;

	this -> n = l + p - 1;

	#if BSPLINE_DEBUG
	std::cout << "Degree (p): " << this -> p << std::endl;
	std::cout << "Number of base functions (n+1): " << this -> n + 1 << std::endl;
	std::cout << "Number of knots (m+1): " << this -> m + 1 << std::endl;
	#endif

	// Resizing the various containers
	this -> knots.resize(this -> m + 1);;
	
	this -> control_point_indices.resize(this -> n + 1);
	for (std::vector<int> & indices : this -> control_point_indices){
		indices.resize(this -> control_point_indices.size());
	}

	// Creating the connectivity table 
	for (int i = 0 ; i < this -> control_point_indices.size(); ++i){
		for(int j = 0; j < this -> control_point_indices.size(); ++j){

			int i_before_wrapping = i;
			int j_before_wrapping = j;

			// Wrapping endpoints to close
			bool wrapping = false;
			if (i > this -> n - this -> p){
				i_before_wrapping -= (this -> n - this -> p + 1);
				wrapping = true;
			}

			// Wrapping endpoints to close
			if (j > this -> n - this -> p){
				j_before_wrapping -= (this -> n - this -> p + 1);
				wrapping = true;
			}

			int index = l * i_before_wrapping + j_before_wrapping;

			this -> control_point_indices[i][j] = index;

			


			if (!wrapping){

				// Inserting the new point in the mesh
				double phi = ( arma::datum::pi * static_cast<double>(i) / l);
				double theta = (2 * arma::datum::pi * static_cast<double>(j ) / l);


				std::cout << "phi : " << phi << " , theta : "  << theta << std::endl;


				arma::vec::fixed<3> coords = {
					std::cos(phi),
					double(j),
					std::sin(phi)
				};

				ControlPoint new_point;
				new_point.set_point_coordinates(coords);
				this -> control_points.push_back(new_point);
			}

			#if BSPLINE_DEBUG
			std::cout << "(i,j): (" << i<< "," << j << "), index = " << index ;
			if (!wrapping){
				std::cout << this -> control_points.back().get_point_coordinates().t();
			}
			else{
				std::cout << std::endl;
			}
			#endif

		}
	}
	assert(this -> control_points.size() == l * l);

	// The uniform knots are generated
	for (int i = 0; i < this -> knots.size(); ++i){
		this -> knots[i] = static_cast<double>(i)/(this -> knots.size() - 1);
	}

	// Creating the control points. 
	
	
}


double BSpline::basis_function(double t, int i,  int j,const std::vector<double> & knots){

	if (j == 0){
		if ( (knots[i] <= t) && (t < knots[i + 1])){
			return 1.;
		}
		else{
			return 0.;
		}
	}
	else{
		
		double a,b;
		if (knots[i + j] != knots[i]){
			a = (t - knots[i])/(knots[i + j] - knots[i]);
		}
		else{
			a = 0.;
		}
		if (knots[i + j + 1] != knots[i + 1]){
			b = ( knots[i + j + 1] - t)/(knots[i + j + 1] - knots[i + 1]);
		}
		else{
			b = 0.;
		}

		return ( a * BSpline::basis_function(t, i,j - 1,knots) 
			+ b * BSpline::basis_function(t, i + 1,j - 1,knots));
	}


}

double BSpline::basis_function(double t, int i,  int j) const{
	return BSpline::basis_function(t,i,j,this -> knots);
}


arma::vec::fixed<3> BSpline::evaluate(double t,double u) const{

	arma::vec::fixed<3> P = {0,0,0};
	for (int i = 0; i <= this -> n ; ++i){
		for (int j = 0; j <= this -> n ; ++j){
			P += (this -> basis_function(t,i,this -> p) 
				* this -> basis_function(u,j,this -> p)
				* this -> control_points[this -> control_point_indices[i][j]].get_point_coordinates());
		}
	}

	return P;

}


double BSpline::get_knot(int i ) const{
	return this -> knots[i];
}


void BSpline::get_domain(double & u_min,double & u_max) const{
	
	u_min = this -> knots[this -> p];
	u_max = this -> knots[this -> n - this -> p];

}


void BSpline::save_control_mesh(std::string partial_path) const{

	PointCloud<PointNormal> pc;
	for (int i = 0; i < this -> control_points.size(); ++i){

		PointNormal p(this -> control_points.at(i).get_point_coordinates(),i);
		pc.push_back(p);

	}
	PointCloudIO<PointNormal>::save_to_obj(pc,partial_path + ".obj");
	PointCloudIO<PointNormal>::save_to_txt(pc,partial_path + ".txt");


}


