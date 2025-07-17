#include <RcppArmadillo.h>
#include <RcppParallel.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace arma;


struct ParallelShannon: public Worker{
  const arma::sp_mat& x;
  RcppParallel::RVector<double> res;

  ParallelShannon(const arma::sp_mat& x, Rcpp::NumericVector res)
      :x(x), res(res){ }
  
  void operator()(std::size_t begin, std::size_t end){
    for (uword i = begin; i < end; i++){
      arma::sp_vec column_val = x.col(i);  
      double col_sum = arma::accu(column_val);  

      if (col_sum <= 0 || column_val.n_nonzero == 0) {  
          res[i] = 0.0;  
          continue;  
      }  

      double tmp_entropy = 0.0;  
      
      for (arma::sp_vec::const_iterator it = column_val.begin(); it != column_val.end(); ++it) {  
          double p = *it / col_sum;  
          tmp_entropy -= p * std::log(p);  
      }  
      
      res[i] = tmp_entropy;  
     
    }
  }
};


// [[Rcpp::export]]
NumericVector cal_shannon(arma::sp_mat x){
  int n = x.n_cols;
  Rcpp::NumericVector entropy(n);

  ParallelShannon parallelshannon(x, entropy);

  parallelFor(0, n, parallelshannon);

  return(entropy);
}



