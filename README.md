# miediff

**miediff** is a C++ header-only library for Mie scattering calculations and analytical Jacobian extraction using Automatic Differentiation (AD).
Designed for researchers in planetary science and atmospheric physics, it provides a bases for aerosol and dust inversion solvers. Miediff calculates precious sensitivity matrices (Jacobians) by using AD.


## Features

* **Exact Jacobians**: Utilizes the `autodiff` library to extract precisious derivatives of scattering properties with respect to refractive index and size distribution parameters.
* **Diverse Size Distributions**: Built-in support for Delta (Single Particle), Log-normal, Gamma, Modified Gamma, Rectangular, and Power-law distributions.
* **Eigen Integration**: Seamlessly outputs results into `Eigen::Matrix` format, making it ready for integration with non-linear optimization solvers (e.g., Levenberg-Marquardt).
* **Precision Quadrature**: Implements Gauss-Legendre, Gauss-Hermite, and Gauss-Laguerre quadrature schemes for fast and stable integration across different distribution types.
* **Header-only**


## Dependencies
* **C++20** or higher.
* **Eigen** (for linear algebra).
* [**autodiff**](https://github.com/tatsuro-iwanaka/autodiff) (the automatic differentiation engine).


## Quick Start

```cpp
#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <Eigen/Dense>

#include "autodiff.hpp"
#include "mie.hpp" 

int main()
{
	// conditions
	int n_theta = 181;
	int n_radius = 5;
	double wavelength = 1.0; // wavelength (micron)
	double r_g = 0.5; // geometric mean radius (micron)
	double sigma_g = std::log(2.0); // geometric standard deviation
	std::complex<double> index(1.53, 0.008); // complex refractive index

	std::cout << "--- Mie Scattering AD Jacobian Test (Log-Normal Distribution) ---" << std::endl;
	std::cout << "Wavelength : " << wavelength << " um" << std::endl;
	std::cout << "r_g        : " << r_g << " um" << std::endl;
	std::cout << "sigma_g    : " << sigma_g << std::endl;
	std::cout << "Ref. Index : " << index << "\n" << std::endl;

	Eigen::VectorXd y_val;
	Eigen::MatrixXd J;

	miediff::DiffFlags diff_flags = {.lnd_r_g = true, .lnd_sigma_g = true, .n_r = true, .n_i = true};

	// calculation
	try
	{
		J = miediff::computeLogNormalMieJacobian(n_theta, n_radius, wavelength, r_g, sigma_g, index, y_val, diff_flags);
	}
	catch (const std::exception& e)
	{
		std::cerr << "[Error] Exception caught during computation: " << e.what() << std::endl;
		return 1;
	}

	// results
	std::cout << std::scientific << std::setprecision(6);

	std::cout << "[ Forward Model Output (y_val) ]" << std::endl;
	std::cout << "Scattering cross section : " << y_val(0) << std::endl;
	std::cout << "Absorption cross section : " << y_val(1) << std::endl;
	std::cout << "Extinction cross section : " << y_val(2) << std::endl;

	std::cout << "Phase Function P(0)      : " << y_val(3) << std::endl;
	std::cout << "Phase Function P(30)     : " << y_val(33) << std::endl;
	std::cout << "Phase Function P(60)     : " << y_val(63) << std::endl;
	std::cout << "Phase Function P(90)     : " << y_val(93) << std::endl;
	std::cout << "Phase Function P(120)    : " << y_val(123) << std::endl;
	std::cout << "Phase Function P(150)    : " << y_val(153) << std::endl;
	std::cout << "Phase Function P(180)    : " << y_val(183) << std::endl;

	std::cout << "\n[ Jacobian Matrix (J) ]" << std::endl;

	std::vector<std::string> labels;
	if (diff_flags.lnd_r_g) labels.push_back("d/dr_g");
	if (diff_flags.lnd_sigma_g) labels.push_back("d/dsigma_g");
	if (diff_flags.n_r) labels.push_back("d/dn_r");
	if (diff_flags.n_i) labels.push_back("d/dn_i");

	std::cout << std::setw(15) << "Parameter";
	for (const auto& label : labels) {
		std::cout << std::setw(15) << label;
	}
	std::cout << "\n" << std::string(15 + 15 * labels.size(), '-') << std::endl;

	std::cout << std::setw(15) << "scs";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(0, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "acs";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(1, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "ecs";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(2, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "P(0)";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(3, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "P(30)";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(33, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "P(60)";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(63, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "P(90)";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(93, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "P(120)";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(123, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "P(150)";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(153, col);
	}
	std::cout << std::endl;

	std::cout << std::setw(15) << "P(180)";
	for (int col = 0; col < J.cols(); ++col)
	{
		std::cout << std::setw(15) << J(183, col);
	}
	std::cout << std::endl;
	

	return 0;
}
