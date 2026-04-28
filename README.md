# miediff

**miediff** is a C++ header-only library for Mie scattering calculations and analytical Jacobian extraction using Automatic Differentiation (AD).
Designed for researchers in planetary science and atmospheric physics, it provides a bases for aerosol and dust inversion solvers. Miediff calculates precious sensitivity matrices (Jacobians) by using AD.


## Key Features

* **Exact Jacobians**: Utilizes the `autodiff` library to extract precisious derivatives of scattering properties with respect to refractive index and size distribution parameters.
* **Diverse Size Distributions**: Built-in support for Delta (Single Particle), Log-normal, Gamma, Modified Gamma, Rectangular, and Power-law distributions.
* **Eigen Integration**: Seamlessly outputs results into `Eigen::Matrix` format, making it ready for integration with non-linear optimization solvers (e.g., Levenberg-Marquardt).
* **Precision Quadrature**: Implements Gauss-Legendre, Gauss-Hermite, and Gauss-Laguerre quadrature schemes for fast and stable integration across different distribution types.
* **Header-only**


## Dependencies
* **C++20** or higher.
* [**Eigen**] (for linear algebra).
* [**autodiff**](https://github.com/tatsuro-iwanaka/autodiff) (the automatic differentiation engine).


## Quick Start

```cpp
#include "mie.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <complex>

int main()
{
  int n_theta = 181; // Number of scattering angles
  double wavelength = 1.0; // Wavelength in [um]
  double r_g = 0.5; // Geometric mean radius [um]
  double sigma_g = std::log(2.0); // Geometric standard deviation
  std::complex<double> index(1.53, 0.008);

  Eigen::VectorXd y; // Output state vector (scattering, absorption and extinction cross sections, Phase Function)
  Eigen::MatrixXd J; // Jacobian matrix (d/dr_g, d/dsigma_g)

  // Compute Jacobians for a Log-normal distribution
  J = computeLogNormalMieJacobian(n_theta, wavelength, r_g, sigma_g, index, y);

  // J(i, 0) is now the exact derivative with respect to r_g
  // J(i, 1) is the exact derivative with respect to sigma_g

  return 0;
}
