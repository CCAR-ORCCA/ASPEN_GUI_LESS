#include <BSpline.hpp>
#include <armadillo>


int main(){


	int l = 11;
	int p = 2;
	
	BSpline spline(l,p);
	int N = 30000;
	arma::mat points(3,N);
	double u_min,u_max;
	spline.get_domain(u_min,u_max);

	spline.save_control_mesh("control_mesh");


	u_max = 1;
	u_min = 0;



	for (int i =0; i < N; ++i){
		arma::vec::fixed<2> coords = (u_max - u_min) * arma::randu<arma::vec>(2) + u_min;
		points.col(i) = spline.evaluate(coords(0),coords(1));
	}
	points.save("points.txt",arma::raw_ascii);


}