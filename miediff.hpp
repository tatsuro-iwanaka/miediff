#include <vector>
#include <numbers>
#include <complex>
#include <cmath>
#include <type_traits>

#include <Eigen/Dense>
#include "autodiff.hpp"

namespace miediff
{

struct DiffFlags
{
	// Delta (Single Particle)
	bool delta_r = true;

	// Log-Normal
	bool lnd_r_g = true;
	bool lnd_sigma_g = true;

	// Rectangular
	bool rect_r_mean = true;
	bool rect_width = true;

	// Gamma
	bool gd_a = true;
	bool gd_b = true;

	// Modified Gamma
	bool mgd_r_c = true;
	bool mgd_alpha = true;
	bool mgd_gamma = true;

	// Power Law
	bool pld_delta = true;
	bool pld_r1 = true;
	bool pld_r2 = true;

	// Refractive Index
	bool n_r = false;
	bool n_i = false;
};

std::vector<std::vector<double>> computeGaussLegendreQuadratureNodeWeight(int);
std::vector<std::vector<double>> computeGaussHermiteQuadratureNodeWeight(int);
std::vector<std::vector<double>> computeGaussLaguerreQuadratureNodeWeight(int);
std::vector<std::vector<double>> computeGeneralizedGaussLaguerreQuadratureNodeWeight(int, double);
template <typename T> T computeSimpsonIntegration(const std::vector<std::vector<T>>&);
template <typename T> void normalizeScatteringPhaseFunction(std::vector<std::vector<T>>&);
template <typename T> void computeMieScattering(int, T, T, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
template <typename T> void computeMieScatteringSizeDistribution(int, T, const std::vector<std::vector<T>>&, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
template <typename T> std::vector<std::vector<std::vector<T>>> generateRectangularSizeDistribution(int, T, T);
template <typename T> std::vector<std::vector<std::vector<T>>> generateLogNormalSizeDistribution(int, T, T);
template <typename T> std::vector<std::vector<std::vector<T>>> generateGammaSizeDistribution(int, T, T);
template <typename T> std::vector<std::vector<std::vector<T>>> generateModifiedGammaSizeDistribution(int, T, T, T);
template <typename T> std::vector<std::vector<std::vector<T>>> generatePowerLawSizeDistribution(int, T, T, T);
template <typename T> void computeDeltaMieScattering(int, T r, T, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
template <typename T> void computeRectangularMieScattering(int, int, T, T, T, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
template <typename T> void computeLogNormalMieScattering(int, int, T, T, T, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
template <typename T> void computeGammaMieScattering(int, int, T, T, T, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
template <typename T> void computeModifiedGammaMieScattering(int, int, T, T, T, T, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
template <typename T> void computePowerLawMieScattering(int, int, T, T, T, T, autodiff::complex<T>, T&, T&, T&, std::vector<std::vector<T>>&);
Eigen::MatrixXd computeDeltaMieJacobian(int, double, double, std::complex<double>, Eigen::VectorXd&, const DiffFlags&);
Eigen::MatrixXd computeLogNormalMieJacobian(int, int, double, double, double, std::complex<double>, Eigen::VectorXd&, const DiffFlags&);
Eigen::MatrixXd computeGammaMieJacobian(int, int, double, double, double, std::complex<double>, Eigen::VectorXd&, const DiffFlags&);
Eigen::MatrixXd computeModifiedGammaMieJacobian(int, int, double, double, double, double, std::complex<double>, Eigen::VectorXd&, const DiffFlags&);
Eigen::MatrixXd computePowerLawMieJacobian(int, int, double, double, double, double, std::complex<double>, Eigen::VectorXd&, const DiffFlags&);

inline std::vector<std::vector<double>> computeGaussLegendreQuadratureNodeWeight(int degree)
{
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int k = 1; k < degree; ++k)
	{
		double k_double = static_cast<double>(k);
		double beta_k = k_double / std::sqrt(4.0 * k_double * k_double - 1.0);

		J(k - 1, k) = beta_k;
		J(k, k - 1) = beta_k;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		// logger::global().fatal("omputeGaussLegendreQuadratureNodeWeight") << "Eigenvalue solution failed for Legendre Jacobi matrix.";
		throw std::runtime_error("Eigenvalue solution failed for Legendre Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i);
		double v_i_1 = eigenvectors(0, i); 
		node_weight[i][1] = 2.0 * v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

inline std::vector<std::vector<double>> computeGaussHermiteQuadratureNodeWeight(int degree)
{
	if (degree <= 0)
	{
		// logger::global().fatal("computeGaussHermiteQuadratureNodeWeight") << "Degree must be positive for Gaussian-Hermite quadrature.";
		throw std::runtime_error("Degree must be positive for Gaussian-Hermite quadrature.");
	}
	
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int k = 1; k < degree; ++k)
	{
		double sub_diag_value = std::sqrt(static_cast<double>(k) / 2.0);
		J(k - 1, k) = sub_diag_value;
		J(k, k - 1) = sub_diag_value;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		// logger::global().fatal("computeGaussHermiteQuadratureNodeWeight") << "Eigenvalue solution failed for Hermite Jacobi matrix.";
		throw std::runtime_error("Eigenvalue solution failed for Hermite Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();

	double sqrt_pi = std::sqrt(std::numbers::pi);
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i); 
		double v_i_1 = eigenvectors(0, i);
		node_weight[i][1] = sqrt_pi * v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

inline std::vector<std::vector<double>> computeGaussLaguerreQuadratureNodeWeight(int degree)
{
	if (degree <= 0)
	{
		// logger::global().fatal("computeGaussLaguerreQuadratureNodeWeight") << "Degree must be positive for Gaussian-Laguerre quadrature.";
		throw std::runtime_error("Degree must be positive for Gaussian-Laguerre quadrature.");
	}
	
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int i = 0; i < degree; ++i)
	{
		double i_double = static_cast<double>(i);
		J(i, i) = 2.0 * i_double + 1.0;
		
		if (i < degree - 1)
		{
			double off_diag_value = i_double + 1.0;
			J(i, i + 1) = off_diag_value; 
			J(i + 1, i) = off_diag_value;
		}
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		// logger::global().fatal("computeGaussLaguerreQuadratureNodeWeight") << "Eigenvalue solution failed for Laguerre Jacobi matrix.";
		throw std::runtime_error("Eigenvalue solution failed for Laguerre Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i); 
		double v_i_1 = eigenvectors(0, i);
		node_weight[i][1] = v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

inline std::vector<std::vector<double>> computeGeneralizedGaussLaguerreQuadratureNodeWeight(int degree, double alpha_prime)
{
	if (degree <= 0)
	{
		// logger::global().fatal("computeGeneralizedGaussLaguerreQuadratureNodeWeight") << "Degree must be positive for Gaussian-Laguerre quadrature.";
		throw std::runtime_error("Degree must be positive for Gaussian-Laguerre quadrature.");
	}
	
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int i = 0; i < degree; ++i)
	{
		double i_double = static_cast<double>(i);
		
		J(i, i) = 2.0 * i_double + 1.0 + alpha_prime;
		
		if (i < degree - 1)
		{
			double k_plus_1 = i_double + 1.0;
			double off_diag_value = std::sqrt(k_plus_1 * (k_plus_1 + alpha_prime));
			
			J(i, i + 1) = off_diag_value; 
			J(i + 1, i) = off_diag_value;
		}
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		// logger::global().fatal("computeGeneralizedGaussLaguerreQuadratureNodeWeight") << "Eigenvalue solution failed for Generalized Laguerre Jacobi matrix.";
		throw std::runtime_error("Eigenvalue solution failed for Generalized Laguerre Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();
	
	double gamma_factor = std::tgamma(alpha_prime + 1.0);
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i);
		double v_i_1 = eigenvectors(0, i); 
		node_weight[i][1] = gamma_factor * v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

template <typename T> inline T computeSimpsonIntegration(const std::vector<std::vector<T>>& f)
{
	int n = f.size();

	if(n % 2 == 0)
	{
		// std::cout << "ERROR: NUMBER OF DATA SHOULD BE ODD." << std::endl;
	}
	
	T result = T(0.0);

	for(int i = 0; i < n; i ++)
	{
		if(i == 0 || i == n - 1)
		{
			result += f[i][1];
		}
		else if(i % 2 == 0)
		{
			result += T(2.0) * f[i][1];
		}
		else if(i % 2 == 1)
		{
			result += T(4.0) * f[i][1];
		}
	}

	result *= (f[1][0] - f[0][0]) / T(3.0);

	return result;
}

template <typename T> inline void normalizeScatteringPhaseFunction(std::vector<std::vector<T>>& f)
{
	using std::sin; using autodiff::sin;

	std::vector<std::vector<T>> phase_function_theta(f.size());

	for(size_t i = 0; i < f.size(); i ++)
	{
		phase_function_theta[i] = {f[i][0], f[i][1] * T(2.0) * T(std::numbers::pi) * sin(f[i][0])};
	}

	T sum =computeSimpsonIntegration(phase_function_theta) / (T(4.0) * T(std::numbers::pi));

	for(size_t i = 0; i < f.size(); i ++)
	{
		f[i][1] /= sum;
	}

	return;
}

template <typename T> inline void computeMieScattering(int n_theta, T radius, T wavelength, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	// Input validation
	if (radius <= T(0.0) || wavelength <= T(0.0) || n_theta < 2)
	{
		scattering_cross_section = T(0.0);
		absorption_cross_section = T(0.0);
		extinction_cross_section = T(0.0);
		phase_function.clear();
		return;
	}

	// サイズパラメータ
	T diameter = T(2.0) * radius;
	T x = T(std::numbers::pi) * diameter / wavelength;

	// 級数上限N_stop
	int nstop;
	if constexpr (std::is_same_v<T, double>)
	{
		nstop = int(std::floor(x + 4.05 * std::cbrt(x) + 2.0));
	}
	else
	{
		nstop = int(std::floor(x.val + 4.05 * std::cbrt(x.val) + 2.0));
	}

	// D_n(z)
	std::vector<autodiff::complex<T>> DD(nstop + 1);

	{
		T threshold = (T(13.78) * index.real() - T(10.8)) * index.real() + T(3.9);
		autodiff::complex<T> z = x * index;
		autodiff::complex<T> zinv(T(0.0), T(0.0));
	
		if (abs(index.imag() * x) < threshold)
		{
			DD[0] = T(1.0) / tan(z);
			zinv  = T(1.0) / z;
			for (int i = 1; i <= nstop; ++i)
			{
				T k = T(double(i));
				auto numerator = k * zinv;
				DD[i] = T(1.0) / (numerator - DD[i - 1]) - numerator;
			}
		}
		else
		{
			zinv = T(2.0) / z;
			auto aj = -(T(nstop) + T(1.5)) * zinv;
			auto alpha_j1 = aj + T(1.0) / ((T(nstop) + T(0.5)) * zinv);
			auto alpha_j2 = aj;
			auto ratio = alpha_j1 / alpha_j2;
			auto runratio = ((T(nstop) + T(0.5)) * zinv) * ratio;

			while (abs(abs(ratio) - T(1.0)) > T(1e-12))
			{
				aj = zinv - aj;
				alpha_j1 = T(1.0) / alpha_j1 + aj;
				alpha_j2 = T(1.0) / alpha_j2 + aj;
				ratio = alpha_j1 / alpha_j2;
				runratio = runratio * ratio;
				zinv = -zinv;
			}

			DD[nstop] = -T(double(nstop)) / z + runratio;
			zinv = T(1.0) / z;

			for (int i = nstop - 1; i >= 0; --i)
			{
				T k = T(double(i + 1));
				auto num = k * zinv;
				DD[i] = num - T(1.0) / (DD[i + 1] + num);
			}
		}
	}

	// Mie係数a_n, b_nと効率Q
	std::vector<autodiff::complex<T>> a(nstop), b(nstop);
	T Qsca = T(0.0);
	T Qext = T(0.0);

	{
		T psi0 = sin(x);
		T psi1 = psi0 / x - cos(x);
	
		autodiff::complex<T> xi0(psi0, -cos(x));
		autodiff::complex<T> xi1(psi1, -(cos(x) / x + sin(x)));

		for (int i = 0; i < nstop; ++i)
		{
			T id = T(double(i + 1));
			a[i] = ((DD[i + 1] / index + id / x) * psi1 - psi0) / ((DD[i + 1] / index + id / x) * xi1 - xi0);
			b[i] = ((DD[i + 1] * index + id / x) * psi1 - psi0) / ((DD[i + 1] * index + id / x) * xi1 - xi0);
			
			T factor0 = T(2.0) * id + T(1.0);
			
			T norm_a = a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
			T norm_b = b[i].real() * b[i].real() + b[i].imag() * b[i].imag();
			
			Qsca = Qsca + factor0 * (norm_a + norm_b);
			Qext = Qext + factor0 * (a[i].real() + b[i].real());
			
			factor0 = (T(2.0) * id + T(1.0)) / x;
			autodiff::complex<T> xi = factor0 * xi1 - xi0;
			xi0 = xi1;
			xi1 = xi;
			
			T psi = factor0 * psi1 - psi0;
			psi0 = psi1;
			psi1 = xi1.real();
		}
	}

	// 無次元効率Qの正規化
	Qsca = T(2.0) * Qsca / (x * x);
	Qext = T(2.0) * Qext / (x * x);
	T Qabs = Qext - Qsca;

	// 断面積
	T area = T(std::numbers::pi) * radius * radius;
	scattering_cross_section = Qsca * area;
	absorption_cross_section = Qabs * area;
	extinction_cross_section = Qext * area;

	// 位相関数
	phase_function.clear();
	phase_function.resize(n_theta);
	T dtheta = T(std::numbers::pi) / T(double(n_theta - 1));

	for (int k = 0; k < n_theta; ++k)
	{
		T theta = dtheta * T(double(k));
		T mu = cos(theta);

		autodiff::complex<T> S1(T(0.0), T(0.0));
		autodiff::complex<T> S2(T(0.0), T(0.0));
		T pi0 = T(0.0);
		T pi1 = T(1.0);

		for (int i = 0; i < nstop; ++i)
		{
			T id = T(double(i + 1));
			T weight = (T(2.0) * id + T(1.0)) / (id + T(1.0)) / id;
			T tau = id * mu * pi1 - (id + T(1.0)) * pi0;

			S1 = S1 + weight * (a[i] * pi1 + b[i] * tau);
			S2 = S2 + weight * (b[i] * pi1 + a[i] * tau);

			T pi2 = ((T(2.0) * id + T(1.0)) * mu * pi1 - (id + T(1.0)) * pi0) / id;
			pi0 = pi1;
			pi1 = pi2;
		}

		T norm_S1 = S1.real() * S1.real() + S1.imag() * S1.imag();
		T norm_S2 = S2.real() * S2.real() + S2.imag() * S2.imag();
		
		T s11 = T(0.5) * (norm_S2 + norm_S1);
		T natural = s11 / (T(std::numbers::pi) * x * x * Qsca);

		phase_function[k] = {theta, natural * T(4.0) * T(std::numbers::pi)};
	}

	return;
}

template <typename T> inline void computeMieScatteringSizeDistribution(int n_theta, T wavelength, const std::vector<std::vector<T>>& node_weight, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	int n_r = node_weight.size();

	T denom = T(0.0);

	absorption_cross_section = T(0.0);
	scattering_cross_section = T(0.0);
	extinction_cross_section = T(0.0);
	
	phase_function.clear();
    phase_function.resize(n_theta, std::vector<T>(2, T(0.0)));

    T dtheta = T(std::numbers::pi) / T(double(n_theta - 1));
    for (int i = 0; i < n_theta; ++i)
    {
        phase_function[i][0] = dtheta * T(double(i));
    }

	for(int i = 0; i < n_r; ++i)
	{
		T scs, acs, ecs;
		std::vector<std::vector<T>> pf;
		T r_i = T(node_weight[i][0]);
		T n_dr = T(node_weight[i][1]);
		computeMieScattering(n_theta, r_i, wavelength, index, scs, acs, ecs, pf);
		absorption_cross_section += acs * n_dr;
		scattering_cross_section += scs * n_dr;
		extinction_cross_section += ecs * n_dr;
		denom += n_dr;

		for(int j = 0; j < n_theta; ++j)
		{
			phase_function[j][1] += pf[j][1] * scs * n_dr;
		}
	}

	absorption_cross_section /= denom;
	scattering_cross_section /= denom;
	extinction_cross_section /= denom;
	
	normalizeScatteringPhaseFunction(phase_function);

	return;
}

template <typename T> inline std::vector<std::vector<std::vector<T>>> generateRectangularSizeDistribution(int n_r, T r_mean, T width)
{
	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || width <= T(0.0))
	{
		// io::global().fatal("generateRectangularSizeDistribution") << "Invalid input parameters (n_r <= 0 or width <= 0).";
		throw std::runtime_error("[generateRectangularSizeDistribution] Invalid input parameters (n_r <= 0 or width <= 0).");
	}

	T r_min = r_mean - T(0.5) * width;
	T r_max = r_mean + T(0.5) * width;

	if (r_min < T(0.0))
	{
		// io::global().fatal("generateRectangularSizeDistribution") << "Rectangular distribution extends to negative radius (r_min < 0). Adjust r_mean or width.";
		throw std::runtime_error("[generateRectangularSizeDistribution] Rectangular distribution extends to negative radius (r_min < 0). Adjust r_mean or width.");
	}

	std::vector<std::vector<double>> node_weight_std = computeGaussLegendreQuadratureNodeWeight(n_r);
	
	T n_r_val = T(1.0) / (r_max - r_min); 

	for (int i = 0; i < n_r; ++i)
	{
		T x_std = T(node_weight_std[i][0]);
		T w_std = T(node_weight_std[i][1]);

		T r = T(0.5) * (r_max - r_min) * x_std + T(0.5) * (r_max + r_min);

		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = n_r_val;
		weight[i][1] = w_std * T(0.5);
	}

	return {size_distribution, weight};
}

template <typename T> inline std::vector<std::vector<std::vector<T>>> generateLogNormalSizeDistribution(int n_r, T r_g, T sigma_g)
{
	using std::log; using autodiff::log;
	using std::exp; using autodiff::exp;
	using std::sqrt; using autodiff::sqrt;

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || r_g <= T(0.0) || sigma_g <= T(0.0))
	{
		if (sigma_g <= T(0.0))
		{
			// io::global().fatal("generateLogNormalSizeDistribution") << "sigma_g must be positive.";
			throw std::runtime_error("[generateLogNormalSizeDistribution] sigma_g must be positive.");
		}
	}

	std::vector<std::vector<double>> node_weight_hermite = computeGaussHermiteQuadratureNodeWeight(n_r);
	
	T ln_rg = log(r_g);
	T sigma_ln = sigma_g;
	
	T sqrt_2_sigma_ln = T(std::numbers::sqrt2) * sigma_ln;

	T weight_prefactor = T(std::numbers::inv_sqrtpi);

	T pdf_norm_factor = T(1.0) / (T(std::numbers::sqrt2) * sigma_ln) * T(std::numbers::inv_sqrtpi);

	for (int i = 0; i < n_r; ++i)
	{
		T u_std = T(node_weight_hermite[i][0]);
		T w_std = T(node_weight_hermite[i][1]);
		
		T r = exp(ln_rg + sqrt_2_sigma_ln * u_std);
		T ln_ratio = log(r / r_g); 
		
		T exponent = -(ln_ratio * ln_ratio) / (T(2.0) * sigma_ln * sigma_ln);

		size_distribution[i][0] = r;
		weight[i][0] = r; 
		size_distribution[i][1] = (pdf_norm_factor / r) * exp(exponent);
		weight[i][1] = w_std * weight_prefactor;
	}

	return {size_distribution, weight};
}

template <typename T> inline std::vector<std::vector<std::vector<T>>> generateGammaSizeDistribution(int n_r, T a, T b)
{
	using std::tgamma; using autodiff::tgamma;
	using std::pow; using autodiff::pow;
	using std::exp; using autodiff::exp;

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || a <= T(0.0) || b <= T(0.0))
	{
		// io::global().fatal("generateGammaSizeDistribution") << "Parameters a and b must be positive.";
		throw std::runtime_error("[generateGammaSizeDistribution] Parameters a and b must be positive.");
	}
	
	if ((T(1.0) - b) / b <= T(0.0))
	{
		// io::global().fatal("generateGammaSizeDistribution") << "(1 - 2b) / b must be positive for convergence. Check parameter b.";
		throw std::runtime_error("[generateGammaSizeDistribution] (1 - 2b) / b must be positive for convergence. Check parameter b.");
	}

	std::vector<std::vector<double>> node_weight_laguerre = computeGaussLaguerreQuadratureNodeWeight(n_r);
	
	T norm_const = T(1.0) / (a * b * tgamma((T(1.0) - T(2.0) * b) / b));

	for (int i = 0; i < n_r; ++i)
	{
		T x_laguerre = T(node_weight_laguerre[i][0]);
		T w_laguerre = T(node_weight_laguerre[i][1]);

		T r = a * b * x_laguerre;
		if (r < T(1e-12))
		{
			r = T(1e-12);
		}

		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = norm_const * pow(r / a / b, (T(1.0) - T(3.0) * b) / b) * exp(-r / a / b);
		weight[i][1] = T(1.0) / tgamma((T(1.0) - T(2.0) * b) / b) * w_laguerre * pow(x_laguerre, (T(1.0) - T(3.0) * b) / b);
	}

	return {size_distribution, weight};
}

template <typename T> inline std::vector<std::vector<std::vector<T>>> generateModifiedGammaSizeDistribution(int n_r, T r_c, T alpha, T gamma)
{
	using std::tgamma; using autodiff::tgamma;
	using std::pow; using autodiff::pow;
	using std::exp; using autodiff::exp;

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || r_c <= T(0.0) || alpha <= T(0.0) || gamma <= T(0.0))
	{
		// io::global().fatal("generateModifiedGammaSizeDistribution") << "Parameters must be positive.";
		throw std::runtime_error("[generateModifiedGammaSizeDistribution] Parameters must be positive.");
	}
	
	// 求積法のパラメータは数値評価が必要なため、実数(double)にキャストして関数へ渡す
	double laguerre_param = get_value((alpha - gamma + T(1.0)) / gamma);
	std::vector<std::vector<double>> node_weight_laguerre = computeGeneralizedGaussLaguerreQuadratureNodeWeight(n_r, laguerre_param);
	
	T norm_const = gamma / (r_c * tgamma((alpha + T(1.0)) / gamma)) * pow(alpha / gamma, (alpha + T(1.0)) / gamma);
	
	for (int i = 0; i < n_r; ++i)
	{
		T x_laguerre = T(node_weight_laguerre[i][0]);
		T w_laguerre = T(node_weight_laguerre[i][1]);

		T r;
		if (x_laguerre == T(0.0))
		{
			r = T(0.0);
		}
		else
		{
			r = r_c * pow((gamma / alpha) * x_laguerre, T(1.0) / gamma);
		}
		
		if (r < T(1e-12))
		{
			r = T(1e-12);
		}

		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = norm_const * pow(r / r_c, alpha) * exp(-alpha / gamma * pow(r / r_c, gamma));
		weight[i][1] = w_laguerre / tgamma((alpha + T(1.0)) / gamma);
	}

	return {size_distribution, weight};
}

template <typename T> inline std::vector<std::vector<std::vector<T>>> generatePowerLawSizeDistribution(int n_r, T pl_delta, T pl_r1, T pl_r2)
{
	using std::log; using autodiff::log;
	using std::exp; using autodiff::exp;
	using std::abs; using autodiff::abs;
	using std::pow; using autodiff::pow;

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || pl_r1 <= T(0.0) || pl_r2 <= T(0.0) || pl_r1 >= pl_r2)
	{
		// io::global().fatal("generatePowerLawSizeDistribution") << "Invalid limits or parameters (r1 >= r2).";
		throw std::runtime_error("[generatePowerLawSizeDistribution] Invalid limits or parameters (r1 >= r2).");
	}

	T ln_r_min = log(pl_r1);
	T ln_r_max = log(pl_r2);

	std::vector<std::vector<double>> node_weight_std = computeGaussLegendreQuadratureNodeWeight(n_r);
	
	T diff = ln_r_max - ln_r_min;
	T sum_limits = ln_r_max + ln_r_min;
	T jacobian = T(0.5) * diff;
	
	T c;
	if (abs(pl_delta - T(1.0)) < T(1e-9))
	{
		c = T(1.0) / diff;
	}
	else
	{
		T term_nu = T(1.0) - pl_delta;
		c = term_nu / (pow(pl_r2, term_nu) - pow(pl_r1, term_nu));
	}

	for (int i = 0; i < n_r; ++i)
	{
		T x_std = T(node_weight_std[i][0]);
		T w_std = T(node_weight_std[i][1]);

		T ln_r = T(0.5) * diff * x_std + T(0.5) * sum_limits;
		T r = exp(ln_r);

		T n_r_val = c * pow(r, -pl_delta);
		T dr = r * w_std * jacobian;
		
		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = n_r_val;
		weight[i][1] = n_r_val * dr;
	}

	return {size_distribution, weight};
}

template <typename T> inline void computeDeltaMieScattering(int n_theta, T r, T wavelength, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	computeMieScattering(n_theta, r, wavelength, index, scattering_cross_section, absorption_cross_section, extinction_cross_section, phase_function);
	normalizeScatteringPhaseFunction(phase_function);
}

template <typename T> inline void computeRectangularMieScattering(int n_theta, int n_r, T r_mean, T width, T wavelength, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	auto results = generateRectangularSizeDistribution(n_r, r_mean, width);
	computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index, scattering_cross_section, absorption_cross_section, extinction_cross_section, phase_function);
}

template <typename T> inline void computeLogNormalMieScattering(int n_theta, int n_r, T r_g, T sigma_g, T wavelength, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	auto results = generateLogNormalSizeDistribution(n_r, r_g, sigma_g);
	computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index, scattering_cross_section, absorption_cross_section, extinction_cross_section, phase_function);
}

template <typename T> inline void computeGammaMieScattering(int n_theta, int n_r, T a, T b, T wavelength, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	auto results = generateGammaSizeDistribution(n_r, a, b);
	computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index, scattering_cross_section, absorption_cross_section, extinction_cross_section, phase_function);
}

template <typename T> inline void computeModifiedGammaMieScattering(int n_theta, int n_r, T r_c, T alpha, T gamma, T wavelength, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	auto results = generateModifiedGammaSizeDistribution(n_r, r_c, alpha, gamma);
	computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index, scattering_cross_section, absorption_cross_section, extinction_cross_section, phase_function);
}

template <typename T> inline void computePowerLawMieScattering(int n_theta, int n_r, T pl_delta, T pl_r1, T pl_r2, T wavelength, autodiff::complex<T> index, T& scattering_cross_section, T& absorption_cross_section, T& extinction_cross_section, std::vector<std::vector<T>>& phase_function)
{
	auto results = generatePowerLawSizeDistribution(n_r, pl_delta, pl_r1, pl_r2);
	computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index, scattering_cross_section, absorption_cross_section, extinction_cross_section, phase_function);
}

inline Eigen::MatrixXd computeDeltaMieJacobian(int n_theta, double wavelength, double r, std::complex<double> index, Eigen::VectorXd& y_val, const DiffFlags& diff_flags = {})
{
	int n_y = 3 + n_theta;
	int n_x = (diff_flags.delta_r ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	y_val.resize(n_y);
	Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(n_y, n_x);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double r_seed, double nr_seed, double ni_seed, int current_col)
	{
		autodiff::dual<double> r_ad(r, r_seed);
		autodiff::dual<double> m_re_ad(index.real(), nr_seed);
		autodiff::dual<double> m_im_ad(index.imag(), ni_seed);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		autodiff::dual<double> scs, acs, ecs;
		std::vector<std::vector<autodiff::dual<double>>> P;

		computeMieScattering(n_theta, r_ad, wl_ad, index_ad, scs, acs, ecs, P);

		if (current_col <= 0)
		{
			y_val(0) = scs.val;
			y_val(1) = acs.val;
			y_val(2) = ecs.val;

			for (int i = 0; i < n_theta; ++i)
			{
				y_val(3 + i) = P[i][1].val;
			}
		}

		if(current_col >= 0)
		{
			Jacobian(0, current_col) = scs.der;
			Jacobian(1, current_col) = acs.der;
			Jacobian(2, current_col) = ecs.der;

			for (int i = 0; i < n_theta; ++i)
			{
				Jacobian(3 + i, current_col) = P[i][1].der;
			}
		}
	};

	if (n_x == 0)
	{
		run_pass(0.0, 0.0, 0.0, -1);

		return Jacobian;
	}

	int col = 0;

	if (diff_flags.delta_r)
	{
		run_pass(1.0, 0.0, 0.0, col++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 1.0, 0.0, col++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 1.0, col++);
	}

	return Jacobian;
}

inline Eigen::MatrixXd computeLogNormalMieJacobian(int n_theta, int n_radius, double wavelength, double r_g, double sigma_g, std::complex<double> index, Eigen::VectorXd& y_val, const DiffFlags& diff_flags = {})
{
	int n_y = 3 + n_theta;
	int n_x = (diff_flags.lnd_r_g ? 1 : 0) + (diff_flags.lnd_sigma_g ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	y_val.resize(n_y);

	Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(n_y, n_x);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double rg_s, double sg_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> rg_ad(r_g, rg_s);
		autodiff::dual<double> sig_ad(sigma_g, sg_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateLogNormalSizeDistribution(n_radius, rg_ad, sig_ad);
		autodiff::dual<double> scs, acs, ecs;
		std::vector<std::vector<autodiff::dual<double>>> P;
		computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad, scs, acs, ecs, P);

		if (col <= 0)
		{
			y_val(0) = scs.val;
			y_val(1) = acs.val;
			y_val(2) = ecs.val;

			for (int i = 0; i < n_theta; ++i)
			{
				y_val(3 + i) = P[i][1].val;
			}
		}

		if (col >= 0)
		{
			Jacobian(0, col) = scs.der;
			Jacobian(1, col) = acs.der;
			Jacobian(2, col) = ecs.der;

			for (int i = 0; i < n_theta; ++i)
			{
				Jacobian(3 + i, col) = P[i][1].der;
			}
		}
	};

	if (n_x == 0)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, -1);
		
		return Jacobian;
	}

	int c = 0;

	if (diff_flags.lnd_r_g)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.lnd_sigma_g)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, c++);
	}


	return Jacobian;
}

inline Eigen::MatrixXd computeRectangularMieJacobian(int n_theta, int n_radius, double wavelength, double r_mean, double width, std::complex<double> index, Eigen::VectorXd& y_val, const DiffFlags& diff_flags = {})
{
	int n_y = 3 + n_theta;
	int n_x = (diff_flags.rect_r_mean ? 1 : 0) + (diff_flags.rect_width ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	y_val.resize(n_y);
	Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(n_y, n_x);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double rm_s, double rw_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> rm_ad(r_mean, rm_s);
		autodiff::dual<double> rw_ad(width, rw_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateRectangularSizeDistribution(n_radius, rm_ad, rw_ad);
		autodiff::dual<double> scs, acs, ecs;
		std::vector<std::vector<autodiff::dual<double>>> P;
		computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad, scs, acs, ecs, P);

		if (col <= 0)
		{
			y_val(0) = scs.val;
			y_val(1) = acs.val;
			y_val(2) = ecs.val;

			for (int i = 0; i < n_theta; ++i)
			{
				y_val(3 + i) = P[i][1].val;
			}
		}

		if (col >= 0)
		{
			Jacobian(0, col) = scs.der;
			Jacobian(1, col) = acs.der;
			Jacobian(2, col) = ecs.der;

			for (int i = 0; i < n_theta; ++i)
			{
				Jacobian(3 + i, col) = P[i][1].der;
			}
		}
	};

	if (n_x == 0)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, -1);
		
		return Jacobian;
	}

	int c = 0;

	if (diff_flags.rect_r_mean)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.rect_width)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, c++);
	}


	return Jacobian;
}

inline Eigen::MatrixXd computeGammaMieJacobian(int n_theta, int n_radius, double wavelength, double a, double b, std::complex<double> index, Eigen::VectorXd& y_val, const DiffFlags& diff_flags = {})
{
	int n_y = 3 + n_theta;
	int n_x = (diff_flags.gd_a ? 1 : 0) + (diff_flags.gd_b ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	y_val.resize(n_y);
	Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(n_y, n_x);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double a_s, double b_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> a_ad(a, a_s);
		autodiff::dual<double> b_ad(b, b_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateGammaSizeDistribution(n_radius, a_ad, b_ad);
		autodiff::dual<double> scs, acs, ecs;
		std::vector<std::vector<autodiff::dual<double>>> P;
		computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad, scs, acs, ecs, P);

		if (col <= 0)
		{
			y_val(0) = scs.val;
			y_val(1) = acs.val;
			y_val(2) = ecs.val;

			for (int i = 0; i < n_theta; ++i)
			{
				y_val(3 + i) = P[i][1].val;
			}
		}

		if (col >= 0)
		{
			Jacobian(0, col) = scs.der; 
			Jacobian(1, col) = acs.der;
			Jacobian(2, col) = ecs.der;

			for (int i = 0; i < n_theta; ++i)
			{
				Jacobian(3 + i, col) = P[i][1].der;
			}
		}
	};

	if (n_x == 0)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, -1);
		
		return Jacobian;
	}

	int c = 0;

	if (diff_flags.gd_a)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.gd_b)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, c++);
	}

	return Jacobian;
}

inline Eigen::MatrixXd computeModifiedGammaMieJacobian(int n_theta, int n_radius, double wavelength, double r_c, double alpha, double gamma, std::complex<double> index, Eigen::VectorXd& y_val, const DiffFlags& diff_flags = {})
{
	int n_y = 3 + n_theta;
	int n_x = (diff_flags.mgd_r_c ? 1 : 0) + (diff_flags.mgd_alpha ? 1 : 0) + (diff_flags.mgd_gamma ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	y_val.resize(n_y);
	Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(n_y, n_x);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double rc_s, double al_s, double ga_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> rc_ad(r_c, rc_s);
		autodiff::dual<double> al_ad(alpha, al_s);
		autodiff::dual<double> ga_ad(gamma, ga_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateModifiedGammaSizeDistribution(n_radius, rc_ad, al_ad, ga_ad);
		autodiff::dual<double> scs, acs, ecs;
		std::vector<std::vector<autodiff::dual<double>>> P;
		computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad, scs, acs, ecs, P);

		if (col <= 0)
		{
			y_val(0) = scs.val;
			y_val(1) = acs.val;
			y_val(2) = ecs.val;

			for (int i = 0; i < n_theta; ++i)
			{
				y_val(3 + i) = P[i][1].val;
			}
		}

		if (col >= 0)
		{
			Jacobian(0, col) = scs.der;
			Jacobian(1, col) = acs.der;
			Jacobian(2, col) = ecs.der;

			for (int i = 0; i < n_theta; ++i)
			{
				Jacobian(3 + i, col) = P[i][1].der;
			}
		}
	};

	if (n_x == 0)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, 0.0, -1);
		return Jacobian;
	}

	int c = 0;

	if (diff_flags.mgd_r_c)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.mgd_alpha)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.mgd_gamma)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, 1.0, c++);
	}


	return Jacobian;
}

inline Eigen::MatrixXd computePowerLawMieJacobian(int n_theta, int n_radius, double wavelength, double pl_delta, double pl_r1, double pl_r2, std::complex<double> index, Eigen::VectorXd& y_val, const DiffFlags& diff_flags = {})
{
	int n_y = 3 + n_theta;
	int n_x = (diff_flags.pld_delta ? 1 : 0) + (diff_flags.pld_r1 ? 1 : 0) + (diff_flags.pld_r2 ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	y_val.resize(n_y);
	Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(n_y, n_x);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double dl_s, double r1_s, double r2_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> dl_ad(pl_delta, dl_s);
		autodiff::dual<double> r1_ad(pl_r1, r1_s);
		autodiff::dual<double> r2_ad(pl_r2, r2_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generatePowerLawSizeDistribution(n_radius, dl_ad, r1_ad, r2_ad);
		autodiff::dual<double> scs, acs, ecs;
		std::vector<std::vector<autodiff::dual<double>>> P;
		computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad, scs, acs, ecs, P);

		if (col <= 0)
		{
			y_val(0) = scs.val;
			y_val(1) = acs.val;
			y_val(2) = ecs.val;

			for (int i = 0; i < n_theta; ++i)
			{
				y_val(3 + i) = P[i][1].val;
			}
		}

		if (col >= 0)
		{
			Jacobian(0, col) = scs.der;
			Jacobian(1, col) = acs.der;
			Jacobian(2, col) = ecs.der;

			for (int i = 0; i < n_theta; ++i)
			{
				Jacobian(3 + i, col) = P[i][1].der;
			}
		}
	};

	if (n_x == 0)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, 0.0, -1);
		return Jacobian;
	}

	int c = 0;

	if (diff_flags.pld_delta)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.pld_r1)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.pld_r2)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, 1.0, c++);
	}
	
	return Jacobian;
}

}
