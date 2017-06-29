#include "ICP.hpp"

ICP::ICP(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source) {

	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;


	this -> register_pc_mrp_multiplicative_partials(30,
	        1e-3,
	        1e-4,
	        true);


}




double ICP::compute_rms_residuals(
    const arma::mat & dcm,
    const arma::vec & x,
    const arma::umat & point_pairs) {

	double J = 0;
	for (arma::uword i = 0; i != point_pairs.n_rows; ++i) {



		J += pow(arma::dot(dcm * this -> pc_source -> get_point_coordinates(point_pairs.row(i)(0))
		                   + x - this -> pc_destination -> get_point_coordinates(point_pairs.row(i)(1)),
		                   this -> pc_destination -> get_point_normal(point_pairs.row(i)(1))), 2 ) ;



		// J += pow(arma::dot(dcm * pc_source.row(point_pairs.row(i)(0)).t()
		//                    + x - pc_destination.row(point_pairs.row(i)(1)).t(),
		//                    destination_normals.row(point_pairs.row(i)(1))), 2 ) ;



	}
	J = std::sqrt(J / point_pairs.n_rows );
	return J;

}











void ICP::register_pc_mrp_multiplicative_partials(
    const unsigned int iterations_max,
    const double rel_tol,
    const double stol,
    const bool pedantic) {

	double J;
	double J_0;
	double J_previous = std::numeric_limits<double>::infinity();
	std::string criterion;


	// The batch estimator is initialized with a zero translation/zero rotation
	arma::vec mrp = {0, 0, 0};
	arma::vec x = {0, 0, 0};

	for (unsigned int iter = 0; iter < iterations_max; ++iter) {

		// The measurement pairs are declared
		arma::umat point_pairs;

		// The pairs are formed

		point_pairs = this -> compute_pairs_closest_compatible_minimum_point_to_plane_dist(
		                  mrp_to_dcm(mrp),
		                  x);

		if (iter == 0) {

			// The initial residuals are computed
			J_0 = this -> compute_rms_residuals(mrp_to_dcm(mrp),
			                                    x,
			                                    point_pairs);
			J = J_0;

		}

		// The matrices of the LS problem are now accumulated
		arma::mat Info_mat = arma::mat(6, 6);
		arma::vec Normal_mat = arma::vec(6);

		for (unsigned int i = 0; i != point_pairs.n_rows; ++i) {

			arma::mat Info_mat_first_rows = arma::mat(3, 6);
			arma::mat Info_mat_second_rows = arma::mat(3, 6);
			arma::mat Info_mat_temp = arma::mat(6, 6);

			arma::vec P_i = this -> pc_source -> get_point_coordinates(point_pairs.row(i)(0));
			arma::vec Q_i = this -> pc_destination -> get_point_coordinates(point_pairs.row(i)(1));
			arma::vec n_i = this -> pc_destination -> get_point_normal(point_pairs.row(i)(1));
			arma::vec Normal_mat_temp = arma::vec(6);

			// The partial derivative of the observation model is computed
			arma::mat H = this -> dGdSigma_multiplicative(mrp, P_i, n_i);

			// The information matrix is constructed by concatenating two 3-by-6 matrices
			Info_mat_first_rows(arma::span(), arma::span(0, 2)) = H.t() * H;
			Info_mat_first_rows(arma::span(), arma::span(3, 5)) = H.t() * n_i.t();
			Info_mat_second_rows(arma::span(), arma::span(0, 2)) = n_i * H ;
			Info_mat_second_rows(arma::span(), arma::span(3, 5)) = n_i * n_i.t();

			Info_mat_temp(arma::span(0, 2), arma::span()) = Info_mat_first_rows;
			Info_mat_temp(arma::span(3, 5), arma::span()) = Info_mat_second_rows;

			// The prefit residuals are computed
			double y_i = arma::dot(n_i.t(), Q_i -  mrp_to_dcm(mrp) * P_i - x );

			// The normal matrix is similarly built
			Normal_mat_temp(arma::span(0, 2)) = H.t() * y_i;
			Normal_mat_temp(arma::span(3, 5)) = n_i * y_i;

			if (i == 0) {
				Info_mat = Info_mat_temp;
				Normal_mat = Normal_mat_temp;
			}
			else {
				Info_mat += Info_mat_temp;
				Normal_mat += Normal_mat_temp;
			}
		}


		// The state deviation [dmrp,dx] is solved for
		arma::vec dX = arma::solve(Info_mat, Normal_mat);
		arma::vec dmrp = {dX(0), dX(1), dX(2)};
		arma::vec dx = {dX(3), dX(4), dX(5)};

		// The state is updated
		mrp = dcm_to_mrp(mrp_to_dcm(dmrp) * mrp_to_dcm(mrp));

		x = x + dx;

		// the mrp is switched to its shadow if need be
		if (arma::norm(mrp) > 1) {
			mrp = - mrp / ( pow(arma::norm(mrp), 2));
		}


		// The postfit residuals are computed
		J = this -> compute_rms_residuals(mrp_to_dcm(mrp),
		                                  x,
		                                  point_pairs);
		if (pedantic == true) {
			std::cout << J << std::endl;
		}


		if ( J / J_0 < rel_tol ) {
			criterion = std::string("r");

			break;
		}

		if ( std::abs(J - J_previous) / J_previous < stol ) {
			criterion = std::string("sr");

			break;
		}

		else if (iter == iterations_max) {
			criterion = std::string("n");

			break;
		}

		// The postfit residuals become the prefit residuals of the next iteration
		J_previous = J;


	}


}


arma::rowvec ICP::dGdSigma_multiplicative(const arma::vec & mrp, const arma::vec & P, const arma::vec & n) {

	arma::rowvec partial = - 4 * P.t() * mrp_to_dcm(mrp).t() * tilde(n);
	return partial;

}



arma::umat ICP::compute_pairs_closest_compatible_minimum_point_to_plane_dist(
    const arma::mat & dcm,
    const arma::mat & x) {

	// arma::mat source_normals_transformed = source_normals * dcm.t();

	// // There will less or as many point pairs as source points
	// arma::umat point_pairs = arma::umat(pc_source.n_rows, 2) ;

	// arma::uword number_of_pairs = 0;

	// for (arma::uword i = 0; i < pc_source.n_rows; ++i) {
	// 	arma::rowvec pc_source_point_transformed = pc_source.row(i) * dcm.t() + x.t();

	// 	double p[3] = {
	// 		pc_source_point_transformed(0),
	// 		pc_source_point_transformed(1),
	// 		pc_source_point_transformed(2)
	// 	};

	// 	unsigned int index_of_closest_destination_point = destination_point_tree -> FindClosestPoint(p);

	// 	if (arma::dot(destination_normals.row(index_of_closest_destination_point),
	// 	              source_normals_transformed.row(i)) > std::sqrt(2) / 2 ) {

	// 		// If the two normals are compatible, the points are matched
	// 		arma::urowvec row = {i, index_of_closest_destination_point};
	// 		point_pairs.row(number_of_pairs) = row;
	// 		++number_of_pairs ;
	// 	}

	// }

	arma::umat point_pairs_clean;

	// point_pairs_clean = point_pairs(arma::span(0, number_of_pairs - 1), arma::span::all);





	return point_pairs_clean;



}

