#ifndef MSC_UTILS_
#define MSC_UTILS_

#include <float.h>  // DBL_EPSILON

#include <cmath>  // std::abs
#include <iostream>

#include "Eigen/Dense"   // Eigen::VectorXd
#include "Eigen/Sparse"  // Eigen::SparseMatrix

/**
 * @file Questo file contiene costanti e metodi da importare facilmente
 *
 */

namespace mcs {

// Numero massimo di iterazioni
constexpr int kMAX_ITER = 20000;

// Tolleranze
constexpr double kTOLL_1 = 1e-4;
constexpr double kTOLL_2 = 1e-6;
constexpr double kTOLL_3 = 1e-8;
constexpr double kTOLL_4 = 1e-10;

/**
 * @brief Condizione di terminazione con residuo scalare
 *
 * @param r residuo
 * @param bnorm norma del termine noto
 * @param toll tolleranza
 * @return true Se ||r|| / ||b|| < toll
 * @return false Altrimenti
 */
inline bool residuoScalare(Eigen::VectorXd& r, double bnorm, double toll) {
  if ((r.norm() / bnorm) < toll) {
    return true;
  }
  return false;
}

/**
 * @brief Condizione di terminazione con residuo scalare
 *
 * @param mat matrice originale
 * @param b termine noto
 * @param x vettore nella k-esima iterazione
 * @param toll tolleranza
 * @return true Se ||r|| / ||b|| < toll
 * @return false Altrimenti
 */
inline bool residuoScalare(Eigen::SparseMatrix<double>& mat, Eigen::VectorXd& b,
                           Eigen::VectorXd& x, double toll) {
  Eigen::VectorXd r = mat * x;
  r = b - r;
  return mcs::residuoScalare(r, b.norm(), toll);
}

// Controlla se due double sono quasi uguali (a meno di EPSILON)
bool dequal(double a, double b, double epsilon = DBL_EPSILON) {
  double res = std::abs(a - b);
  // find max without abs (?)
  double max = std::max(std::abs(a), std::abs(b));
  return res <= epsilon * max;
}

}  // namespace mcs

#endif  // MSC_UTILS_