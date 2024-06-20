#ifndef MCS_SOLVER_
#define MCS_SOLVER_

#include "Eigen/Dense"   // Eigen::VectorXd
#include "Eigen/Sparse"  // Eigen::SparseMatrix

/**
 * @brief File per la solver class
 *
 */

namespace mcs {

class Solver {
 public:
  Solver(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b) : mat(A), b(b){};

  std::size_t getIteration() { return iter; }
  Eigen::VectorXd getResult() { return x; }

  void setToll(double newToll) { toll = newToll; }
  void setMaxIter(std::size_t newMax) { maxIter = newMax; }

  void solve() {
    prepare();
    while (checkIter()) {
      iterate();
    }
  }

 protected:
  Eigen::VectorXd r;     // vettore residuo
  Eigen::VectorXd x;     // vettore soluzione
  double bnorm;          // calcolo norma b solo una volta
  std::size_t iter = 0;  // counter interazioni

  Eigen::SparseMatrix<double>& mat;  // ref a matrice A
  Eigen::VectorXd& b;                // ref a vettore termini noti b

  std::size_t maxIter = mcs::kMAX_ITER;  // max numero di iterazioni consentite
  double toll = mcs::kTOLL_3;            // tolleranza calcolo terminazione

  /**
   * @brief Calcola il criterio di arresto
   *
   * @return true arresto
   * @return false continuo
   */
  bool checkIter() {
    if (iter > maxIter) return false;
    ++iter;
    return !mcs::residuoScalare(r, bnorm, toll);
  }

  /**
   * @brief base per inizializzazione parametri prima del solve
   *
   */
  virtual void prepare() {
    bnorm = b.norm();  // calcolo norma vettore noto una volta
    r = b;             // inizializzo r a valore di b per far partire while
    x = b;
    x.setConstant(0);  // inizializzo x a 0
  }

  /**
   * @brief Esegue un iterazione dell'algoritmo
   *
   */
  virtual void iterate() = 0;
};

}  // namespace mcs

#endif