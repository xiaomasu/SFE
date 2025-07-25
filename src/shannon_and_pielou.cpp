#include <RcppArmadillo.h>
#include <RcppParallel.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace arma;


struct ParallelShannon: public Worker{
  const arma::sp_mat& x;
  RcppParallel::RMatrix<double> res;

  ParallelShannon(const arma::sp_mat& x, Rcpp::NumericMatrix res)
      :x(x), res(res) { }
  
  void operator()(std::size_t begin, std::size_t end){
    for (uword i = begin; i < end; i++){
      arma::sp_vec column_val = x.col(i);  
      double col_sum = arma::accu(column_val);  
      uword num_features = column_val.n_nonzero;

      if (col_sum <= 0 || column_val.n_nonzero == 0) {  
          res(i,0)= 0.0;
	  res(i,1) = 0.0;  
          continue;  
      }  

      double shannon = 0.0;  
      
      for (arma::sp_vec::const_iterator it = column_val.begin(); it != column_val.end(); ++it) {  
          double p = *it / col_sum;  
          shannon -= p * std::log(p);  
      }  
      
      double pielou = 0.0;
      if(column_val.n_nonzero > 1){
	      pielou = shannon / (std::log2(num_features));
      }
      res(i,0) = shannon;
      res(i,1) = pielou;  
     
    }
  }
};


// [[Rcpp::export]]
NumericMatrix cal_shannon(arma::sp_mat x){
  int n = x.n_cols;
  Rcpp::NumericMatrix res(n,2);

  ParallelShannon parallelshannon(x, res);

  parallelFor(0, n, parallelshannon);

  return res;
 
}



