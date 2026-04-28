#include <vector>
#include <numbers>
#include <complex>
#include <cmath>
#include <type_traits>

#include <Eigen/Dense>
#include "autodiff.hpp"

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

template <typename T> inline void normalizeScatteringPhaseFunction(std::vector<std::vector<T>> &f)
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

inline Eigen::MatrixXd computeDeltaMieJacobian(int n_theta, double wavelength, double r, std::complex<double> index, Eigen::VectorXd& y_val)
{
	Eigen::MatrixXd Jacobian;

	autodiff::dual<double> wl_ad(wavelength, 0.0); 
	autodiff::complex<autodiff::dual<double>> index_ad = index; 

	int n_y = 3 + n_theta; // 状態ベクトルの次元数 (scs, acs, ecs, P(theta))
	int n_x = 1; // パラメータの数 (r_g, sigma_g)

	y_val.resize(n_y);
	Jacobian.resize(n_y, n_x);

	// r に対するヤコビアン (第0列)
	autodiff::dual<double> r_ad(r, 1.0);
	
	autodiff::dual<double> scs, acs, ecs;
	std::vector<std::vector<autodiff::dual<double>>> P;

	computeMieScattering(n_theta, r_ad, wl_ad, index_ad, scs, acs, ecs, P);

	y_val(0) = scs.val; Jacobian(0, 0) = scs.der;
	y_val(1) = acs.val; Jacobian(1, 0) = acs.der;
	y_val(2) = ecs.val; Jacobian(2, 0) = ecs.der;

	for(int i = 0; i < n_theta; ++i)
	{
		y_val(3 + i) = P[i][1].val;
		Jacobian(3 + i, 0) = P[i][1].der;
	}

	return Jacobian;
}

inline Eigen::MatrixXd computeLogNormalMieJacobian(int n_theta, double wavelength, double r_g, double sigma_g, std::complex<double> index, Eigen::VectorXd& y_val)
{
	Eigen::MatrixXd Jacobian;

	autodiff::dual<double> wl_ad(wavelength, 0.0); 
	autodiff::complex<autodiff::dual<double>> index_ad = index;

	int n_y = 3 + n_theta;
	int n_x = 2;

	y_val.resize(n_y);
	Jacobian.resize(n_y, n_x);

	// r_g に対するヤコビアン (第0列)
	autodiff::dual<double> rg_ad_1(r_g, 1.0);
	autodiff::dual<double> sig_ad_1(sigma_g, 0.0);
	
	autodiff::dual<double> scs1, acs1, ecs1;
	std::vector<std::vector<autodiff::dual<double>>> P1;

	auto sd1 = generateLogNormalSizeDistribution(n_theta, rg_ad_1, sig_ad_1);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd1[1], index_ad, scs1, acs1, ecs1, P1);

	y_val(0) = scs1.val;
	y_val(1) = acs1.val;
	y_val(2) = ecs1.val;

	Jacobian(0, 0) = scs1.der;
	Jacobian(1, 0) = acs1.der;
	Jacobian(2, 0) = ecs1.der;

	for(int i = 0; i < n_theta; ++i)
	{
		y_val(3 + i) = P1[i][1].val;
		Jacobian(3 + i, 0) = P1[i][1].der;
	}

	// sigma_g に対するヤコビアン (第1列)
	autodiff::dual<double> rg_ad_2(r_g, 0.0);
	autodiff::dual<double> sig_ad_2(sigma_g, 1.0);
	
	autodiff::dual<double> scs2, acs2, ecs2;
	std::vector<std::vector<autodiff::dual<double>>> P2;

	auto sd2 = generateLogNormalSizeDistribution(n_theta, rg_ad_2, sig_ad_2);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd2[1], index_ad, scs2, acs2, ecs2, P2);

	Jacobian(0, 1) = scs2.der;
	Jacobian(1, 1) = acs2.der;
	Jacobian(2, 1) = ecs2.der;

	for(int i = 0; i < n_theta; ++i)
	{
		Jacobian(3 + i, 1) = P2[i][1].der;
	}

	return Jacobian;
}

inline Eigen::MatrixXd computeRectangularMieJacobian(int n_theta, double wavelength, double r_mean, double width, std::complex<double> index, Eigen::VectorXd& y_val)
{
	Eigen::MatrixXd Jacobian;

	autodiff::dual<double> wl_ad(wavelength, 0.0); 
	autodiff::complex<autodiff::dual<double>> index_ad = index; 

	int n_y = 3 + n_theta;
	int n_x = 2;

	y_val.resize(n_y);
	Jacobian.resize(n_y, n_x);

	autodiff::dual<double> r_mean_ad_1(r_mean, 1.0);
	autodiff::dual<double> width_ad_1(width, 0.0);
	
	autodiff::dual<double> scs1, acs1, ecs1;
	std::vector<std::vector<autodiff::dual<double>>> P1;

	auto sd1 = generateRectangularSizeDistribution(n_theta, r_mean_ad_1, width_ad_1);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd1[1], index_ad, scs1, acs1, ecs1, P1);

	y_val(0) = scs1.val;
	y_val(1) = acs1.val;
	y_val(2) = ecs1.val;

	Jacobian(0, 0) = scs1.der;
	Jacobian(1, 0) = acs1.der;
	Jacobian(2, 0) = ecs1.der;

	for(int i = 0; i < n_theta; ++i)
	{
		y_val(3 + i) = P1[i][1].val;
		Jacobian(3 + i, 0) = P1[i][1].der;
	}

	autodiff::dual<double> r_mean_ad_2(r_mean, 0.0);
	autodiff::dual<double> width_ad_2(width, 1.0);
	
	autodiff::dual<double> scs2, acs2, ecs2;
	std::vector<std::vector<autodiff::dual<double>>> P2;

	auto sd2 = generateRectangularSizeDistribution(n_theta, r_mean_ad_2, width_ad_2);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd2[1], index_ad, scs2, acs2, ecs2, P2);

	Jacobian(0, 1) = scs2.der;
	Jacobian(1, 1) = acs2.der;
	Jacobian(2, 1) = ecs2.der;

	for(int i = 0; i < n_theta; ++i)
	{
		Jacobian(3 + i, 1) = P2[i][1].der;
	}

	return Jacobian;
}

inline Eigen::MatrixXd computeGammaMieJacobian(int n_theta, double wavelength, double a, double b, std::complex<double> index, Eigen::VectorXd& y_val)
{
	Eigen::MatrixXd Jacobian;

	autodiff::dual<double> wl_ad(wavelength, 0.0); 
	autodiff::complex<autodiff::dual<double>> index_ad = index; 

	int n_y = 3 + n_theta;
	int n_x = 2;

	y_val.resize(n_y);
	Jacobian.resize(n_y, n_x);

	autodiff::dual<double> a_ad_1(a, 1.0);
	autodiff::dual<double> b_ad_1(b, 0.0);
	
	autodiff::dual<double> scs1, acs1, ecs1;
	std::vector<std::vector<autodiff::dual<double>>> P1;

	auto sd1 = generateGammaSizeDistribution(n_theta, a_ad_1, b_ad_1);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd1[1], index_ad, scs1, acs1, ecs1, P1);

	y_val(0) = scs1.val;
	y_val(1) = acs1.val;
	y_val(2) = ecs1.val;

	Jacobian(0, 0) = scs1.der;
	Jacobian(1, 0) = acs1.der;
	Jacobian(2, 0) = ecs1.der;


	for(int i = 0; i < n_theta; ++i)
	{
		y_val(3 + i) = P1[i][1].val;
		Jacobian(3 + i, 0) = P1[i][1].der;
	}
	
	autodiff::dual<double> a_ad_2(a, 0.0);
	autodiff::dual<double> b_ad_2(b, 1.0);
	
	autodiff::dual<double> scs2, acs2, ecs2;
	std::vector<std::vector<autodiff::dual<double>>> P2;

	auto sd2 = generateGammaSizeDistribution(n_theta, a_ad_2, b_ad_2);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd2[1], index_ad, scs2, acs2, ecs2, P2);

	Jacobian(0, 1) = scs2.der;
	Jacobian(1, 1) = acs2.der;
	Jacobian(2, 1) = ecs2.der;

	for(int i = 0; i < n_theta; ++i)
	{
		Jacobian(3 + i, 1) = P2[i][1].der;
	}

	return Jacobian;
}

inline Eigen::MatrixXd computeModifiedGammaMieJacobian(int n_theta, double wavelength, double r_c, double alpha, double gamma, std::complex<double> index, Eigen::VectorXd& y_val)
{
	Eigen::MatrixXd Jacobian;

	autodiff::dual<double> wl_ad(wavelength, 0.0); 
	autodiff::complex<autodiff::dual<double>> index_ad = index; 

	int n_y = 3 + n_theta;
	int n_x = 3; 

	y_val.resize(n_y);
	Jacobian.resize(n_y, n_x);

	autodiff::dual<double> rc_ad_1(r_c, 1.0);
	autodiff::dual<double> alpha_ad_1(alpha, 0.0);
	autodiff::dual<double> gamma_ad_1(gamma, 0.0);
	
	autodiff::dual<double> scs1, acs1, ecs1;
	std::vector<std::vector<autodiff::dual<double>>> P1;

	auto sd1 = generateModifiedGammaSizeDistribution(n_theta, rc_ad_1, alpha_ad_1, gamma_ad_1);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd1[1], index_ad, scs1, acs1, ecs1, P1);

	y_val(0) = scs1.val;
	y_val(1) = acs1.val;
	y_val(2) = ecs1.val;

	Jacobian(0, 0) = scs1.der;
	Jacobian(1, 0) = acs1.der;
	Jacobian(2, 0) = ecs1.der;

	for(int i = 0; i < n_theta; ++i)
	{
		y_val(3 + i) = P1[i][1].val;
		Jacobian(3 + i, 0) = P1[i][1].der;
	}

	autodiff::dual<double> rc_ad_2(r_c, 0.0);
	autodiff::dual<double> alpha_ad_2(alpha, 1.0);
	autodiff::dual<double> gamma_ad_2(gamma, 0.0);
	
	autodiff::dual<double> scs2, acs2, ecs2;
	std::vector<std::vector<autodiff::dual<double>>> P2;

	auto sd2 = generateModifiedGammaSizeDistribution(n_theta, rc_ad_2, alpha_ad_2, gamma_ad_2);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd2[1], index_ad, scs2, acs2, ecs2, P2);

	Jacobian(0, 1) = scs2.der;
	Jacobian(1, 1) = acs2.der;
	Jacobian(2, 1) = ecs2.der;

	for(int i = 0; i < n_theta; ++i)
	{
		Jacobian(3 + i, 1) = P2[i][1].der;
	}

	autodiff::dual<double> rc_ad_3(r_c, 0.0);
	autodiff::dual<double> alpha_ad_3(alpha, 0.0);
	autodiff::dual<double> gamma_ad_3(gamma, 1.0);
	
	autodiff::dual<double> scs3, acs3, ecs3;
	std::vector<std::vector<autodiff::dual<double>>> P3;

	auto sd3 = generateModifiedGammaSizeDistribution(n_theta, rc_ad_3, alpha_ad_3, gamma_ad_3);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd3[1], index_ad, scs3, acs3, ecs3, P3);

	Jacobian(0, 2) = scs3.der;
	Jacobian(1, 2) = acs3.der;
	Jacobian(2, 2) = ecs3.der;

	for(int i = 0; i < n_theta; ++i)
	{
		Jacobian(3 + i, 2) = P3[i][1].der;
	}

	return Jacobian;
}

inline Eigen::MatrixXd computePowerLawMieJacobian(int n_theta, double wavelength, double pl_delta, double pl_r1, double pl_r2, std::complex<double> index, Eigen::VectorXd& y_val)
{
	Eigen::MatrixXd Jacobian;

	autodiff::dual<double> wl_ad(wavelength, 0.0); 
	autodiff::complex<autodiff::dual<double>> index_ad = index; 

	int n_y = 3 + n_theta;
	int n_x = 3;

	y_val.resize(n_y);
	Jacobian.resize(n_y, n_x);

	autodiff::dual<double> delta_ad_1(pl_delta, 1.0);
	autodiff::dual<double> r1_ad_1(pl_r1, 0.0);
	autodiff::dual<double> r2_ad_1(pl_r2, 0.0);
	
	autodiff::dual<double> scs1, acs1, ecs1;
	std::vector<std::vector<autodiff::dual<double>>> P1;

	auto sd1 = generatePowerLawSizeDistribution(n_theta, delta_ad_1, r1_ad_1, r2_ad_1);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd1[1], index_ad, scs1, acs1, ecs1, P1);

	y_val(0) = scs1.val; 
	y_val(1) = acs1.val; 
	y_val(2) = ecs1.val;

	Jacobian(0, 0) = scs1.der;
	Jacobian(1, 0) = acs1.der;
	Jacobian(2, 0) = ecs1.der;

	for(int i = 0; i < n_theta; ++i)
	{
		y_val(3 + i) = P1[i][1].val;
		Jacobian(3 + i, 0) = P1[i][1].der;
	}

	autodiff::dual<double> delta_ad_2(pl_delta, 0.0);
	autodiff::dual<double> r1_ad_2(pl_r1, 1.0);
	autodiff::dual<double> r2_ad_2(pl_r2, 0.0);
	
	autodiff::dual<double> scs2, acs2, ecs2;
	std::vector<std::vector<autodiff::dual<double>>> P2;

	auto sd2 = generatePowerLawSizeDistribution(n_theta, delta_ad_2, r1_ad_2, r2_ad_2);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd2[1], index_ad, scs2, acs2, ecs2, P2);

	Jacobian(0, 1) = scs2.der;
	Jacobian(1, 1) = acs2.der;
	Jacobian(2, 1) = ecs2.der;

	for(int i = 0; i < n_theta; ++i)
	{
		Jacobian(3 + i, 1) = P2[i][1].der;
	}

	autodiff::dual<double> delta_ad_3(pl_delta, 0.0);
	autodiff::dual<double> r1_ad_3(pl_r1, 0.0);
	autodiff::dual<double> r2_ad_3(pl_r2, 1.0);
	
	autodiff::dual<double> scs3, acs3, ecs3;
	std::vector<std::vector<autodiff::dual<double>>> P3;

	auto sd3 = generatePowerLawSizeDistribution(n_theta, delta_ad_3, r1_ad_3, r2_ad_3);
	computeMieScatteringSizeDistribution(n_theta, wl_ad, sd3[1], index_ad, scs3, acs3, ecs3, P3);

	Jacobian(0, 2) = scs3.der;
	Jacobian(1, 2) = acs3.der;
	Jacobian(2, 2) = ecs3.der;

	for(int i = 0; i < n_theta; ++i)
	{
		Jacobian(3 + i, 2) = P3[i][1].der;
	}

	return Jacobian;
}
