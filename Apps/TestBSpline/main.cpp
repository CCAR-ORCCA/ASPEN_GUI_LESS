#include <BSpline.hpp>
#include <armadillo>


int main(){

	int n_u,n_v,p_u,p_v;

	n_u = 7;
	n_v = 7;

	p_u = 2;
	p_v = 2;

	std::vector<double> knots = {};

	BSpline spline(n_u,n_v,p_u,p_v,knots,knots,BSpline::Type::Closed);
	double u_min,u_max,v_min,v_max;
	spline.get_domain(u_min,v_min,u_max,v_max);

	std::cout << "Domain : u = [" << u_min << " , " << u_max  << "], : v = [" << v_min << " , " << v_max  << "]\n";

	int N = 30000;
	arma::mat points(3,N);

	spline.save_control_mesh("control_mesh");


	for (int i =0; i < N; ++i){
		arma::vec::fixed<2> coords = arma::randu<arma::vec>(2);
		points.col(i) = spline.evaluate(
			(u_max - u_min) * coords(0) + u_min,
			(v_max - v_min) * coords(1) + v_min);
	}

	points.save("points.txt",arma::raw_ascii);


}