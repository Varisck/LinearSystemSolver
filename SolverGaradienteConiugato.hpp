#ifndef MCS_SOLVER_GRADIENTE_CONGIUNTO_
#define MCS_SOLVER_GRADIENTE_CONGIUNTO_

#include <iostream>

#include "./Solver.hpp"  // msc::Solver
#include "Eigen/Dense"   // Eigen::VectorXd
#include "Eigen/Sparse"  // Eigen::SparseMatrix

/**
 * @brief File per la solver class
 *
 */

namespace mcs {

class SolverGaradienteConiugato : public mcs::Solver {
 public:
  SolverGaradienteConiugato(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b)
      : mcs::Solver(A, b){};

 private:
  Eigen::VectorXd d;  // vettore aggiuntivo
  double alpha;
  double beta;
  double dDotAd;

  void prepare() override {
    mcs::Solver::prepare();
    alpha = 0.0;
    beta = 0.0;
    dDotAd = 0.0;
    d = b;  // modo veloce per settare size
  }

  void iterate() override {
    r = b - (mat * x);
    dDotAd = d.dot(mat * d);
    alpha = d.dot(r) / dDotAd;
    x = x + (alpha * d);
    r = b - (mat * x);
    beta = d.dot(mat * r) / dDotAd;
    d = r - (beta * d);
  };
};

}  // namespace mcs

#endif