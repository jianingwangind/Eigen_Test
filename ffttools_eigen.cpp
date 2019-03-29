#include "ffttools_eigen.hpp"

namespace eco
{
	vfc::float32_t mat_sum_f_eigen(const EigenMatDynamic f_matrixIn)
	{
		return f_matrixIn.sum();
	}

	std::vector<EigenMatDynamic> cvMat2EigenMat(const cv::Mat& f_incvMat)
	{
		std::vector<EigenMatDynamic> res;
		if (f_incvMat.channels() == 1)
		{
			EigenMatDynamic temp(f_incvMat.rows, f_incvMat.cols);
			cv::cv2eigen(f_incvMat, temp);
			res.push_back(temp);
		}
		else if (f_incvMat.channels() == 2)
		{
			std::vector<cv::Mat> cvtemp;
			cv::split(f_incvMat, cvtemp);
			
			EigenMatDynamic tempa(f_incvMat.rows, f_incvMat.cols), tempb(f_incvMat.rows, f_incvMat.cols);
			cv::cv2eigen(cvtemp[0], tempa);
			cv::cv2eigen(cvtemp[1], tempb);
			res.push_back(tempa);
			res.push_back(tempb);
		}	
		return res;
	}

	cv::Mat EigenMat2cvMat(const std::vector<EigenMatDynamic> f_matrixIn)
	{
		cv::Mat tempa, tempb, res;
		cv::eigen2cv(f_matrixIn[0], tempa); 	
		std::vector<cv::Mat> cvtemp;
		cvtemp.push_back(tempa);

		if (f_matrixIn.size() == 2)
		{
			cv::eigen2cv(f_matrixIn[1], tempb);
			cvtemp.push_back(tempb);
		}
		cv::merge(cvtemp, res);

		return res;
	}

	EigenMatDynamic real_eigen(const std::vector<EigenMatDynamic> f_matrixIn)
	{
		return f_matrixIn[0];
	}

	std::vector<EigenMatDynamic> mat_conj_eigen(const std::vector<EigenMatDynamic> f_matrixIn)
	{
		std::vector<EigenMatDynamic> res;
		res.push_back(f_matrixIn[0]);
		EigenMatDynamic temp;
		temp = f_matrixIn[1] * (-1);
		res.push_back(temp);

		return res;
	}

	std::vector<EigenMatDynamic> real2complex_eigen(const std::vector<EigenMatDynamic> f_matrixIn)
	{
		if (f_matrixIn.size() == 2)
			return f_matrixIn;
		EigenMatDynamic zeros;
		zeros.setZero();
		std::vector<EigenMatDynamic> results;
		results.push_back(f_matrixIn[0]);
		results.push_back(zeros);

		return results;
	}

	EigenMatDynamic correlation_eigen(const EigenMatDynamic& f_src, const EigenMatDynamic& f_kernel)
	{
		EigenMatDynamic f_dst(f_src.rows(), f_src.cols());
		vfc::CPoint anchor(f_kernel.cols() / 2, f_kernel.rows() / 2);
		for (vfc::int32_t i = 0; i < f_src.rows(); i++)
		{
			for (vfc::int32_t j = 0; j < f_src.cols(); j++)
			{
				vfc::float32_t sum = 0;
				for (vfc::int32_t k = 0; k < f_kernel.rows(); k++)
				{
					for (vfc::int32_t l = 0; l < f_kernel.cols(); l++)
					{
						if ((i + k - anchor.y()) < 0 || (j + l - anchor.x()) < 0 ||
							(i + k - anchor.y()) > (f_src.rows() - 1) || (j + l - anchor.x()) > (f_src.cols() - 1))
							continue;
						sum += f_kernel(k, l) * f_src(i + k - anchor.y(), j + l - anchor.x());
					}
				}
				f_dst(i, j) = sum;
			}
		}
		return f_dst;
	}

	std::vector<EigenMatDynamic> complexConvolution_eigen(const std::vector<EigenMatDynamic> f_matrixInA,
		const std::vector<EigenMatDynamic> f_matrixInB,
		const bool valid)
	{
		std::vector<EigenMatDynamic> a_temp, res, a(2), b;
		a[0].conservativeResize(f_matrixInA[0].rows() + f_matrixInB[0].rows() - 1, f_matrixInA[0].cols() + f_matrixInB[0].cols() - 1);
		a[1].conservativeResize(f_matrixInA[0].rows() + f_matrixInB[0].rows() - 1, f_matrixInA[0].cols() + f_matrixInB[0].cols() - 1);
		a[0].setZero();
		a[1].setZero();
		if (f_matrixInA.size() == 1)
			a_temp = real2complex_eigen(f_matrixInA);
		else if (f_matrixInA.size() == 2)
			a_temp = f_matrixInA;
		else if (f_matrixInA.size() > 2)
			assert(0 && "error: a_input's channel dimensions error!");
		if (f_matrixInB.size() == 1)
			b = real2complex_eigen(f_matrixInB);
		else if (f_matrixInA.size() == 2)
			b = f_matrixInB;
		else if (f_matrixInA.size() > 2)
			assert(0 && "error: b_input's channel dimensions error!");

		vfc::CPoint pos(f_matrixInB[0].cols() / 2, f_matrixInB[0].rows() / 2);

		a[0].block(f_matrixInB[0].rows() - 1 - pos.y(),
			f_matrixInB[0].cols() - 1 - pos.x(),
			f_matrixInA[0].rows(),
			f_matrixInA[0].cols()) = a_temp[0];
		a[1].block(f_matrixInB[0].rows() - 1 - pos.y(),
			f_matrixInB[0].cols() - 1 - pos.x(),
			f_matrixInA[0].rows(),
			f_matrixInA[0].cols()) = a_temp[1];

		EigenMatDynamic r, i, r1, r2, i1, i2;
		r1 = correlation_eigen(a[0], b[0].reverse());
		r2 = correlation_eigen(a[1], b[1].reverse());
		i1 = correlation_eigen(a[0], b[1].reverse());
		i2 = correlation_eigen(a[1], b[0].reverse());

		r = r1 - r2;
		i = i1 + i2;
		res.push_back(r);
		res.push_back(i);

		if (valid)
		{
			if (f_matrixInB[0].cols() > f_matrixInA[0].cols() || f_matrixInB[0].rows() > f_matrixInA[0].rows())
			{
				Eigen::MatrixXf zero(0, 0);
				zero.setZero();
				std::vector<EigenMatDynamic> temp;
				temp.push_back(zero);
				temp.push_back(zero);
				return temp;
			}
			else
			{
				std::vector<EigenMatDynamic> temp;
				EigenMatDynamic tempa, tempb;
				tempa = res[0].block(f_matrixInB[0].rows() - 1, f_matrixInB[0].cols() - 1,
					f_matrixInA[0].rows() - f_matrixInB[0].cols() + 1, f_matrixInA[0].cols() - f_matrixInB[0].cols() + 1);
				tempb = res[1].block(f_matrixInB[0].rows() - 1, f_matrixInB[0].cols() - 1,
					f_matrixInA[0].rows() - f_matrixInB[0].cols() + 1, f_matrixInA[0].cols() - f_matrixInB[0].cols() + 1);
				temp.push_back(tempa);
				temp.push_back(tempb);

				return temp;
			}
		}
		else
		{
			return res;
		}
	}

	std::vector<EigenMatDynamic> complexDotMultiplication_eigen(const std::vector<EigenMatDynamic> f_matrixInA, const std::vector<EigenMatDynamic> f_matrixInB)
	{
		std::vector<EigenMatDynamic> temp_A = f_matrixInA, temp_B = f_matrixInB;
		if (f_matrixInA.size() == 1)
			temp_A = real2complex_eigen(f_matrixInA);
		if (f_matrixInB.size() == 1)
			temp_B = real2complex_eigen(f_matrixInB);

		std::vector<EigenMatDynamic> res;
		EigenMatDynamic real_temp, imag_temp;

		real_temp = temp_A[0].array() * temp_B[0].array() - temp_A[1].array() * temp_B[1].array();
		imag_temp = temp_A[0].array() * temp_B[1].array() + temp_A[1].array() * temp_B[0].array();

		res.push_back(real_temp);
		res.push_back(imag_temp);

		return res;
	}

	std::vector<EigenMatDynamic> complexDotDevision_eigen(const std::vector<EigenMatDynamic> f_matrixInA, const std::vector<EigenMatDynamic> f_matrixInB)
	{
		std::vector<EigenMatDynamic> res;
		EigenMatDynamic divisor = f_matrixInB[0].array() * f_matrixInB[0].array() + f_matrixInB[1].array() * f_matrixInB[1].array();
		EigenMatDynamic real_temp, imag_temp;
		real_temp = (f_matrixInA[0].array() * f_matrixInB[0].array() + f_matrixInA[1].array() * f_matrixInB[1].array()) / divisor.array();
		imag_temp = (f_matrixInA[1].array() * f_matrixInB[0].array() - f_matrixInA[0].array() * f_matrixInB[1].array()) / divisor.array();
		res.push_back(real_temp);
		res.push_back(imag_temp);

		return res;
	}

	std::vector<EigenMatDynamic> complexMatrixMultiplication_eigen(const std::vector<EigenMatDynamic> f_matrixInA, const std::vector<EigenMatDynamic> f_matrixInB)
	{
		if (f_matrixInA[0].cols() != f_matrixInB[0].rows())
			assert(0 && "error: a and b size unmatched!");

		std::vector<EigenMatDynamic> res;
		EigenMatDynamic real_temp, imag_temp;

		real_temp = f_matrixInA[0] * f_matrixInB[0] - f_matrixInA[1] * f_matrixInB[1];
		imag_temp = f_matrixInA[0] * f_matrixInB[1] + f_matrixInA[1] * f_matrixInB[0];

		res.push_back(real_temp);
		res.push_back(imag_temp);

		return res;
	}
}
