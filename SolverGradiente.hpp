#ifndef MCS_SOLVER_GRADIENTE_
#define MCS_SOLVER_GRADIENTE_

#include "./Solver.hpp"  // msc::Solver
#include "Eigen/Dense"   // Eigen::VectorXd
#include "Eigen/Sparse"  // Eigen::SparseMatrix

/**
 * @brief File per la solver class
 *
 */

namespace mcs {

class SolverGradiente : public mcs::Solver {
 public:
  SolverGradiente(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b)
      : mcs::Solver(A, b){};

 private:
  double alpha;

  void prepare() override {
    mcs::Solver::prepare();
    alpha = 0.0;
  }

  void iterate() override {
    r = b - (mat * x);
    alpha = r.dot(r) / r.dot(mat * r);
    x = x + (alpha * r);
  };
};

}  // namespace mcs

#endif