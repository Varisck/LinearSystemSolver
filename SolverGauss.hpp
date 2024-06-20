#ifndef MCS_SOLVER_GAUSS_
#define MCS_SOLVER_GAUSS_

#include <iostream>

#include "./Solver.hpp"  // msc::Solver
#include "./Utils.h"     // msc::dequal
#include "Eigen/Dense"   // Eigen::VectorXd
#include "Eigen/Sparse"  // Eigen::SparseMatrix

/**
 * @brief File per la solver class
 *
 */

namespace mcs {

class SolverGauss : public mcs::Solver {
 public:
  SolverGauss(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b)
      : mcs::Solver(A, b){};

 private:
  Eigen::SparseMatrix<double> P;  // matrice tirangolare inferiore di mat
  Eigen::VectorXd y;              // vettore per calcolare y = P^{-1} * r

  /**
   * @brief Controlla che gli elementi della diagonale siano != 0
   *  utilizza funzione msc::dequal per controllare double == double
   *
   * @param mat matrice sparsa
   * @return true gli elementi della diagonale sono diversi da 0
   * @return false esiste almeno un elemento sulla diagonale uguale a 0
   */
  inline bool checkDiagonal(Eigen::SparseMatrix<double>& mat) {
    for (std::size_t i = 0; i < mat.cols(); ++i) {
      if (mcs::dequal(mat.coeff(i, i), 0.0)) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Calcola una forward substitution (mat * x) = b
   *
   * @param mat Matrice sparsa triangolare inferiore, importante si suppone
   * che gli elemnti sulla diagnole != 0
   * @param b vettore di termini noti
   * @param y vettore soluzione
   * @return true non ci sono stati errori
   * @return false ci sono stati errori
   */
  inline void forward(Eigen::SparseMatrix<double>& mat, Eigen::VectorXd& x,
                      Eigen::VectorXd& b) {
    x.setConstant(0);
    x[0] = b[0] / mat.coeff(0, 0);
    for (std::size_t i = 1; i < mat.cols(); ++i) {
      x[i] = (b[i] - (mat.row(i) * x)) / mat.coeff(i, i);
    }
  }

  void prepare() override {
    mcs::Solver::prepare();
    P = mat.triangularView<Eigen::Lower>();  // prende triangolare inf di mat
    assert(!checkDiagonal(P) && "Errore trovato 0 sulla diagnoale di A");
    // if (!checkDiagonal(P))
    //   iter = maxIter + 1;  // controlla che non ci sono 0 nella diagonale
    y = Eigen::VectorXd(b.size());
    y.setConstant(0);  // inizializza y
  }

  void iterate() override {
    r = b - (mat * x);
    forward(P, y, r);  // sostituzione in avanti Py = r
    x = x + y;
  };
};

}  // namespace mcs

#endif