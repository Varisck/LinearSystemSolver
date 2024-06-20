#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "./Gauss.h"
#include "./Gradiente.h"
#include "./Jacobi.h"
#include "Eigen/Dense"                          // Eigen::VectorXd
#include "Eigen/Sparse"                         // Eigen::SparseMatrix<>
#include "Eigen/unsupported/Eigen/SparseExtra"  // Eigen::loadMarket

// ===== classes =====
#include "./SolverGaradienteConiugato.hpp"
#include "./SolverGauss.hpp"
#include "./SolverGradiente.hpp"
#include "./SolverJacobi.hpp"

inline void output(std::string filename, std::vector<std::size_t> v1,
                   std::vector<double> v2) {
  std::ofstream outputFile(filename);
  if (!outputFile.is_open()) {
    std::cerr << "Error opening the file." << std::endl;
    return;
  }
  for (int i = 0; i < v2.size(); ++i) {
    outputFile << "N: " << i << " it: " << v1[i] << ", time: " << v2[i]
               << std::setprecision(9) << std::endl;
  }
}

int main() {
  std::vector<std::string> files{"./matrici/vem2.mtx"};
  std::vector<double> tolls{mcs::kTOLL_1, mcs::kTOLL_2, mcs::kTOLL_3,
                            mcs::kTOLL_4};

  std::vector<double> times;
  std::vector<std::size_t> iterazioni;

  for (auto file : files) {
    std::cout << "Matrice: " << file << std::endl;
    Eigen::SparseMatrix<double> mat;
    if (Eigen::loadMarket(mat, file)) {
      Eigen::VectorXd b(mat.rows());
      Eigen::VectorXd x(mat.rows());
      x.setConstant(1);
      b = mat * x;
      for (int t = 0; t < tolls.size(); ++t) {
        std::cout << "Tolleranza: " << tolls[t] << std::endl;
        std::cout << "Starting Jacobi" << std::endl;
        // Jacobi
        for (int i = 0; i < 10; ++i) {
          mcs::SolverJacobi js(mat, b);
          js.setToll(tolls[t]);
          auto start = std::chrono::high_resolution_clock::now();
          js.solve();
          auto end = std::chrono::high_resolution_clock::now();
          iterazioni.push_back(js.getIteration());
          times.push_back(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count());
          times[i] *= 1e-9;
        }
        std::string outputName =
            file + "Jacobi-Toll" + std::to_string(t) + ".txt";
        output(outputName, iterazioni, times);
        times.clear();
        iterazioni.clear();

        std::cout << "Starting Gauss" << std::endl;
        // gauss
        for (int i = 0; i < 1; ++i) {
          mcs::SolverGauss gs(mat, b);
          gs.setToll(tolls[t]);
          auto start = std::chrono::high_resolution_clock::now();
          gs.solve();
          auto end = std::chrono::high_resolution_clock::now();
          iterazioni.push_back(gs.getIteration());
          times.push_back(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count());
          times[i] *= 1e-9;
        }

        outputName = file + "Gauss-Toll" + std::to_string(t) + ".txt";
        output(outputName, iterazioni, times);
        times.clear();
        iterazioni.clear();

        std::cout << "Starting Gradiente" << std::endl;
        // grad
        for (int i = 0; i < 10; ++i) {
          mcs::SolverGradiente gs(mat, b);
          gs.setToll(tolls[t]);
          auto start = std::chrono::high_resolution_clock::now();
          gs.solve();
          auto end = std::chrono::high_resolution_clock::now();
          iterazioni.push_back(gs.getIteration());
          times.push_back(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count());
          times[i] *= 1e-9;
        }

        outputName = file + "grad-Toll" + std::to_string(t) + ".txt";
        output(outputName, iterazioni, times);
        times.clear();
        iterazioni.clear();

        std::cout << "Starting Gradiente Congiunto" << std::endl;
        // grad Acc
        for (int i = 0; i < 10; ++i) {
          mcs::SolverGaradienteConiugato gcs(mat, b);
          gcs.setToll(tolls[t]);
          auto start = std::chrono::high_resolution_clock::now();
          gcs.solve();
          auto end = std::chrono::high_resolution_clock::now();
          iterazioni.push_back(gcs.getIteration());
          times.push_back(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count());
          times[i] *= 1e-9;
        }

        outputName = file + "Gauss-GradC" + std::to_string(t) + ".txt";
        output(outputName, iterazioni, times);
        times.clear();
        iterazioni.clear();
      }
    } else {
      std::cout << "ERRORE CARICAMENTO MATRICE!";
      return 1;
    }
  }
  return 0;
}