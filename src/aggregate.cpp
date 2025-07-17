// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <map>
#include <cmath> 

struct CountShannonWorker : public RcppParallel::Worker {
    // Inputs (read-only)
    const arma::umat& index_matrix;
    const arma::vec& values_vector;
    
    // Output (write-only, to distinct elements)
    arma::vec& output_entropy;

    // Constructor
    CountShannonWorker(const arma::umat& im, const arma::vec& vv, arma::vec& oe)
        : index_matrix(im), values_vector(vv), output_entropy(oe) {}

    // Parallel execution operator
    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            
            // 1. Frequency counting for the current row
            std::map<double, int> row_counts;
            int total_valid_elements = 0;

            for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                // Get 1-based index from the input matrix
                arma::uword r_index = index_matrix(i, j);
                
                // Safety check: ensure index is within the bounds of the values_vector
                if (r_index > 0 && r_index <= values_vector.n_elem) {
                    // Convert to 0-based index for C++/Armadillo
                    arma::uword c_index = r_index - 1;
                    row_counts[values_vector(c_index)]++;
                    total_valid_elements++;
                }
            }

            // 2. Calculate Shannon Entropy for the current row
            double current_entropy = 0.0;
            if (total_valid_elements > 1) {
                
                // C++11 compatible loop for iterating over a map
                for (std::map<double, int>::const_iterator it = row_counts.begin(); it != row_counts.end(); ++it) {
                    // int val = it->first; // Key
                    int count = it->second; // Value
                    
                    if (count > 0) {
                        double p = static_cast<double>(count) / total_valid_elements;
                        current_entropy -= p * std::log(p);
                    }
                }
            }
            
            // 3. Store the result in the output vector (thread-safe)
            output_entropy(i) = current_entropy;
        }
    }
};

struct ColumnMergerWorker : public RcppParallel::Worker {
    const arma::umat index_matrix;
    const arma::sp_mat& x_sp_mat;
    arma::vec& res;

    ColumnMergerWorker(const arma::umat& im, const arma::sp_mat& gsm, arma::vec& res)
        : index_matrix(im), x_sp_mat(gsm), res(res) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
           arma::sp_vec column_val = sum(x_sp_mat.cols(index_matrix.row(i)-1), 1);
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
arma::vec merge_columns_parallel(const arma::umat& index_matrix, const arma::sp_mat& x_sp_mat) {
    if (index_matrix.n_rows == 0) {
        return arma::vec();
    }

    //arma::uword out_rows = x_sp_mat.n_rows;
    arma::uword out_cols = index_matrix.n_rows;

    arma::vec result(out_cols);

    ColumnMergerWorker worker(index_matrix, x_sp_mat, result);

    RcppParallel::parallelFor(0, out_cols, worker);

    return result;
}


// [[Rcpp::export]]
arma::vec calculate_count_shannon(const arma::umat& index_matrix, const arma::vec& values_vector) {
    // Handle empty input
    if (index_matrix.n_rows == 0) {
        return arma::vec();
    }
    
    // Prepare the output vector
    arma::vec result_vector(index_matrix.n_rows, arma::fill::zeros);

    // Create the worker instance
    CountShannonWorker worker(index_matrix, values_vector, result_vector);
    
    // Execute the parallel loop
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    
    return result_vector;
}

