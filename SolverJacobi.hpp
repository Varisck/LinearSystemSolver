#ifndef MCS_SOLVER_JACOBI_
#define MCS_SOLVER_JACOBI_

#include <iostream>

#include "./Solver.hpp"  // msc::Solver
#include "Eigen/Dense"   // Eigen::VectorXd
#include "Eigen/Sparse"  // Eigen::SparseMatrix

/**
 * @brief File per la solver class
 *
 */

namespace mcs {

class SolverJacobi : public mcs::Solver {
 public:
  SolverJacobi(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b)
      : mcs::Solver(A, b){};

 private:
  Eigen::VectorXd p;  // vettore della diagonale di mat

  void prepare() override {
    mcs::Solver::prepare();
    p = mat.diagonal();
    p = p.array().inverse();  // calcola 1/p per ogni posizione
  }

  void iterate() override {
    r = b - (mat * x);
    x = x + r.cwiseProduct(p);
  };
};

}  // namespace mcs

#endif