#include <BSpline.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <PointCloudIO.hpp>

#define BSPLINE_DEBUG 1


BSpline::BSpline(int n_u,int n_v,int p_u,int p_v,
	std::vector<double> knots_u,
	std::vector<double> knots_v,
	int type){

	this -> type = type;
	
	// Setting degrees
	this -> p_u = p_u;
	this -> p_v = p_v;

	// Settings number of basis functions
	this -> n_u = n_u;
	this -> n_v = n_v;

	// Setting knots
	this -> knots_u = knots_u;
	this -> knots_v = knots_v;

	// Resizing the knots containers and generating uniform knots
	// if no knots were provided
	if (knots_u.size() == 0 || type == BSpline::Type::Closed){
		this -> m_u = this -> n_u + this -> p_u + 1;
		this -> knots_u.resize(this -> m_u + 1);
		for (int i = 0; i < this -> knots_u.size(); ++i){
			this -> knots_u[i] = static_cast<double>(i)/(this -> knots_u.size() - 1);
		}
	}
	else{
		this -> m_u = static_cast<int>(knots_u.size() ) - 1;
	}

	if (knots_v.size() == 0 || type == BSpline::Type::Closed){
		this -> m_v = this -> n_v + this -> p_v + 1;
		this -> knots_v.resize(this -> m_v + 1);
		for (int i = 0; i < this -> knots_v.size(); ++i){
			this -> knots_v[i] = static_cast<double>(i)/(this -> knots_v.size() - 1);
		}
	}
	else{
		this -> m_v = static_cast<int>(knots_v.size() ) - 1;

	}

	this -> control_point_indices.resize(this -> n_u + 1);

	for (std::vector<int> & indices : this -> control_point_indices){
		indices.resize(this -> n_v + 1);
	}



	if (type != BSpline::Type::Closed){
		
		// Creating the connectivity table 
		for (int i = 0 ; i < this -> n_u + 1; ++i){
			for(int j = 0; j < this -> n_v + 1; ++j){

				ControlPoint new_point;

				arma::vec::fixed<3> coords = {
					static_cast<double>(j),
					static_cast<double>(i),
					1 - std::sqrt( std::pow(i - this -> n_u/2.,2) + std::pow(j - this -> n_v/2.,2))
				};

				new_point.set_point_coordinates(coords);
				this -> control_point_indices[i][j] = this -> control_points.size();
				this -> control_points.push_back(new_point);


			}
		}
	}
	else{

		// Creating the connectivity table 
		for (int i = 0 ; i < this -> n_u + 1; ++i){
			for(int j = 0; j < this -> n_v + 1; ++j){

				int i_before_wrapping = i;
				int j_before_wrapping = j;

			// Wrapping endpoints to close
				bool wrapping_u = false;
				bool wrapping_v = false;

				if (i > this -> n_u - this -> p_u){
					i_before_wrapping -= (this -> n_u - this -> p_u + 1);
					wrapping_u = true;
				}

			// Wrapping endpoints to close

				// if (j == this -> n){
				// 	// j_before_wrapping -= (this -> n - this -> p + 1);
				// 	// wrapping_v = true;
				// }

				if (wrapping_u){

					int index = (this -> n_v + 1 ) * i_before_wrapping + j_before_wrapping;
					this -> control_point_indices[i][j] = index;

				}
				else if (wrapping_v){

					// int index = (this -> n_v + 1) * i_before_wrapping + j_before_wrapping;
					// this -> control_point_indices[i][j] = index;

				}
				else{

					ControlPoint new_point;
					double angle = std::floor(static_cast<double>(i) / (this -> n_u - this -> p_u + 1) * 8) * arma::datum::pi/4;
					
					arma::vec::fixed<3> coords = {
						static_cast<double>(j),
						std::cos(angle),
						std::sin(angle)
						
					};


					new_point.set_point_coordinates(coords);
					this -> control_point_indices[i][j] = this -> control_points.size();
					this -> control_points.push_back(new_point);
				}

			}
		}


	}


}









double BSpline::basis_function(double t, int i,  int p,const std::vector<double> & knots){


	if (p == 0){
		if ( (knots[i] <= t) && (t < knots[i + 1])){
			return 1.;
		}
		else{
			return 0.;
		}
	}
	else{
		


		double a,b;

		if (knots[i + p] != knots[i]){
			a = (t - knots[i])/(knots[i + p] - knots[i]);
		}

		else{
			a = 0.;
		}

		if (knots[i + p + 1] != knots[i + 1]){
			b = ( knots[i + p + 1] - t)/(knots[i + p + 1] - knots[i + 1]);
		}

		else{
			b = 0.;
		}


		return ( a * BSpline::basis_function(t, i,p - 1,knots) 
			+ b * BSpline::basis_function(t, i + 1,p - 1,knots));
	}


}



arma::vec::fixed<3> BSpline::evaluate(double u,double v) const{

	arma::vec::fixed<3> P = {0,0,0};

	for (int i = 0; i < this -> n_u  + 1; ++i){
		for (int j = 0; j < this -> n_v  + 1; ++j){

			P += (BSpline::basis_function(u,i,this -> p_u,this -> knots_u) 
				* BSpline::basis_function(v,j,this -> p_v,this -> knots_v)
				* this -> control_points[this -> control_point_indices[i][j]].get_point_coordinates());
		}
	}

	return P;

}




void BSpline::check_if_clamped(bool & is_open_at_start,bool & is_open_at_end,
	int p,const std::vector<double> & knots){

	is_open_at_start = false;
	for (int i = 0; i < p + 1; ++i){
		if (knots[i] != knots[0]){
			is_open_at_start = true;
			break;
		}
	}

	is_open_at_end = false;
	for (int i = 0; i < p + 1; ++i){
		if (knots[static_cast<int>(knots.size()) - 1 - i] != knots.back()){
			is_open_at_end = true;

			break;
		}
	}

}

void BSpline::get_domain(double & u_min,double & v_min,double & u_max,double & v_max) const{


	if (this -> type == BSpline::Type::Closed){

		u_min = this -> knots_u[this -> p_u];
		u_max = this -> knots_u[this -> m_u - this -> p_u];

		bool is_open_at_start,is_open_at_end;

		BSpline::check_if_clamped(is_open_at_start,is_open_at_end,this -> p_v,this -> knots_v);
		if (is_open_at_start){
			v_min = this -> knots_v[this -> p_v];
		}
		else{
			v_min = 0;
		}
		if (is_open_at_end){
			v_max = this -> knots_v[this -> m_v - this -> p_v];
		}
		else{
			v_max = 1;
		}

	}
	else{
	// u
		bool is_open_at_start,is_open_at_end;
		BSpline::check_if_clamped(is_open_at_start,is_open_at_end,this -> p_u,this -> knots_u);

		if (is_open_at_start){
			u_min = this -> knots_u[this -> p_u];
		}
		else{
			u_min = 0;
		}
		if (is_open_at_end){
			u_max = this -> knots_u[this -> m_u - this -> p_u];
		}
		else{
			u_max = 1;
		}

		BSpline::check_if_clamped(is_open_at_start,is_open_at_end,this -> p_v,this -> knots_v);
		if (is_open_at_start){
			v_min = this -> knots_v[this -> p_v];
		}
		else{
			v_min = 0;
		}
		if (is_open_at_end){
			v_max = this -> knots_v[this -> m_v - this -> p_v];
		}
		else{
			v_max = 1;
		}
	}


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


