#ifndef FFTTOOLS_EIGEN_HPP
#define FFTTOOLS_EIGEN_HPP

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "vfc/core/vfc_core_all.hpp"

#include <assert.h>

namespace eco
{
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatDynamic;

	std::vector<EigenMatDynamic> cvMat2EigenMat(const cv::Mat& f_incvMat);

	cv::Mat EigenMat2cvMat(const std::vector<EigenMatDynamic> f_matrixIn);

	EigenMatDynamic real_eigen(const std::vector<EigenMatDynamic> f_matrixIn);

	vfc::float32_t mat_sum_f_eigen(const EigenMatDynamic f_matrixIn);

	std::vector<EigenMatDynamic> mat_conj_eigen(const std::vector<EigenMatDynamic> f_matrixIn);

	std::vector<EigenMatDynamic> real2complex_eigen(const std::vector<EigenMatDynamic> f_matrixIn);

	EigenMatDynamic correlation_eigen(const EigenMatDynamic& f_src, const EigenMatDynamic& f_kernel);

	std::vector<EigenMatDynamic> complexConvolution_eigen(const std::vector<EigenMatDynamic> f_matrixInA,
		const std::vector<EigenMatDynamic> f_matrixInB,
		const bool valid = 0);

	std::vector<EigenMatDynamic> complexDotMultiplication_eigen(const std::vector<EigenMatDynamic> f_matrixInA, const std::vector<EigenMatDynamic> f_matrixInB);

	std::vector<EigenMatDynamic> complexDotDevision_eigen(const std::vector<EigenMatDynamic> f_matrixInA, const std::vector<EigenMatDynamic> f_matrixInB);

	std::vector<EigenMatDynamic> complexMatrixMultiplication_eigen(const std::vector<EigenMatDynamic> f_matrixInA, const std::vector<EigenMatDynamic> f_matrixInB);
}

#endif
