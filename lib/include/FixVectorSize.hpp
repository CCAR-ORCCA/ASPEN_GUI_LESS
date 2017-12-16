#ifndef HEADER_FIXVECTORSIZE
#define HEADER_FIXVECTORSIZE

#include <armadillo>

namespace boost { 
    namespace numeric { 
        namespace odeint {

template <>
            struct is_resizeable<arma::vec>{
                typedef boost::true_type type;
                const static bool value = type::value;
            };

template <>
            struct same_size_impl<arma::vec, arma::vec>{
                static bool same_size(const arma::vec & x, const arma::vec& y){
                    return x.n_rows == y.n_rows;  
                }
            };

template<>
            struct resize_impl<arma::vec, arma::vec>{
                static void resize(arma::vec &v1, const arma::vec& v2){
                    v1.reshape(v2.n_rows,1);     
                }
            };

        } 
    } 
} // namespace boost::numeric::odeint
#endif