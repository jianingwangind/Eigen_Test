#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Matrix3Dynamicf;

int main()
{
	//Eigen::MatrixXd m(2, 2);
	//m(0, 0) = 3;
	//m(1, 0) = 2.5;
	//m(0, 1) = -1;
	//m(1, 1) = m(1, 0) + m(0, 1);
	//std::cout << m << std::endl;

	//Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 3);
	Eigen::Matrix3d m = Eigen::Matrix3d::Random();
	//m = (m + Eigen::MatrixXd::Constant(3, 3, 1.2)) * 50;
	m = (m + Eigen::Matrix3d::Constant(1.2)) * 50;
	std::cout << "m= " << std::endl << m << std::endl << m(1) << std::endl;
	
	//Eigen::VectorXd v(3); // (column) vector with 3 unintialized entries
	//v << 1, 2, 3; // comma-initializer
	Eigen::Vector3d v(1, 2, 3);
	std::cout << "m * v = " << std::endl << m * v << std::endl;

	Eigen::RowVector2i r(1, 2), s(3, 4);
	Eigen::RowVector4i t;
	Eigen::Matrix2i u;
	t << r, s;
	u << r, s;
	std::cout << "concatenated row vector is: " << t << std::endl;
	std::cout << "concatenated matrix is: " << std::endl << u << std::endl;

	Eigen::Matrix<int, 3, 4, Eigen::ColMajor> Acolmajor;
	Acolmajor << 8, 2, 2, 9,
				 9, 1, 4, 4,
				 3, 5, 4, 5;
	std::cout << "The matrix A: " << std::endl;
	std::cout << Acolmajor << std::endl << std::endl;

	std::cout << "In memory (column major): " << std::endl;
	for (int i = 0; i < Acolmajor.size(); i++)
		std::cout << *(Acolmajor.data() + i) << " ";
	std::cout << std::endl;

	Eigen::Matrix<int, 3, 4, Eigen::RowMajor> Arowmajor = Acolmajor;
	std::cout << "In memory (row major): " << std::endl;
	for (int i = 0; i < Arowmajor.size(); i++)
		std::cout << *(Arowmajor.data() + i) << " ";
	std::cout << std::endl;

	Eigen::Matrix3f m52_;
	m52_.row(0) << 1, 2, 3;
	m52_.block(1, 0, 2, 2) << 4, 5, 7, 8;
	m52_.col(2).head(3) << 100, 6, 9;
	m52_.col(2).tail(1) << 999;
	std::cout << "blockwise initialization: " << std::endl << m52_ << std::endl;

	Eigen::Array33f a1 = Eigen::Array33f::Zero();
	Eigen::ArrayXf  a2 = Eigen::ArrayXf::Zero(3);
	std::cout << "zeros vector: " << std::endl << a2 << std::endl;
	Eigen::ArrayXXf a3 = Eigen::ArrayXXf::Ones(3, 4);
	std::cout << "ones array: " << std::endl << a3 << std::endl;

	Eigen::Matrix3i i65_ = Eigen::Matrix3i::Identity();
	std::cout << "identity matrix: " << std::endl << i65_ << std::endl;

	Eigen::VectorXf v68_ = Eigen::VectorXf::LinSpaced(10, 0.0, 80.0);
	Eigen::ArrayXf a68_ = v68_;
	std::cout << "linspaced vector: " << std::endl << a68_.col(0) * 0.1 << std::endl << a68_.col(0).sin() << std::endl;

	const int size = 6;
	Eigen::MatrixXd mat1(size, size);
	mat1.topLeftCorner(size / 2, size / 2) = Eigen::MatrixXd::Zero(size / 2, size / 2);
	mat1.topRightCorner(size / 2, size / 2) = Eigen::MatrixXd::Identity(size / 2, size / 2);
	mat1.bottomLeftCorner(size / 2, size / 2) = Eigen::MatrixXd::Identity(size / 2, size / 2);
	mat1.bottomRightCorner(size / 2, size / 2) = Eigen::MatrixXd::Zero(size / 2, size / 2);
	std::cout << "4 blockwise mat1" << std::endl << mat1 << std::endl << std::endl;

	Eigen::MatrixXd mat2(size, size);
	//mat2.topLeftCorner(size / 2, size / 2).setZero();
	mat2.block(0, 0, size / 2, size / 2).setZero();
	mat2.topRightCorner(size / 2, size / 2).setIdentity();
	mat2.bottomLeftCorner(size / 2, size / 2).setIdentity();
	mat2.col(size / 2).setLinSpaced(size, 10, 20);
	mat2.bottomRightCorner(size / 2, size / 2).setRandom();
	mat2.setZero();
	Eigen::MatrixXd demo(6, 3);
	demo.setOnes();
	mat2.block(0, 2, 6, 3) = demo;
	std::cout << "4 blockwise mat2" << std::endl << mat2 << std::endl << std::endl;
	mat2.transposeInPlace();
	std::cout << "mat2 transpose" << std::endl << mat2 << std::endl << std::endl;

	Eigen::MatrixXd mat3(size, size);
	mat3 << Eigen::MatrixXd::Zero(size / 2, size / 2), Eigen::MatrixXd::Identity(size / 2, size / 2),
			Eigen::MatrixXd::Identity(size / 2, size / 2), Eigen::MatrixXd::Zero(size / 2, size / 2);
	std::cout << "4 blockwise mat3" << std::endl << mat3 << std::endl << std::endl;
	mat3.conservativeResize(Eigen::NoChange_t::NoChange, 5);
	std::cout << "resized mat3" << std::endl << mat3 << std::endl << std::endl;

	Eigen::MatrixXcf c98_ = Eigen::MatrixXcf::Random(2, 2);
	Eigen::MatrixXcf c98_conj_ = c98_.conjugate();

	// cv::Mat to Eigen::Matrix
	cv::Mat koala = cv::imread("C:\\Users\\wja4hi\\Desktop\\Koala.jpg");
	Eigen::Matrix<uchar, -1, -1, Eigen::RowMajor> koala_eigen_(koala.rows, koala.cols);
	std::vector<cv::Mat> koala_container_;
	cv::split(koala, koala_container_);
	cv::cv2eigen(koala_container_[0], koala_eigen_);
	std::cout << "koala_eigen_: " << (unsigned)koala_eigen_(0, 0)<< std::endl;
	std::cout << "koala: " << (unsigned)koala.at<cv::Vec3b>(0, 0)[0] << std::endl;

	cv::Mat cv_eigen_ = cv::Mat::eye(3, 3, CV_32FC2);
	std::vector<cv::Mat> cv_eigen_container_;
	cv::split(cv_eigen_, cv_eigen_container_);

	Eigen::Matrix<float, -1, -1, Eigen::RowMajor> cv_eigen_c1_(cv_eigen_.rows, cv_eigen_.cols);
	Eigen::Matrix<float, -1, -1, Eigen::RowMajor> cv_eigen_c2_(cv_eigen_.rows, cv_eigen_.cols);
	cv::cv2eigen(cv_eigen_container_[0], cv_eigen_c1_);
	cv::cv2eigen(cv_eigen_container_[1], cv_eigen_c2_);
	std::cout << "cv_eigen_c1_: " << cv_eigen_c1_(0, 0) << std::endl;
	std::cout << "cv_eigen_c2_: " << cv_eigen_c2_(0, 0) << std::endl;
	cv_eigen_c1_(0, 0) = 100;
	std::cout << "cv_eigen_container_[0]: " << cv_eigen_container_[0].at<float>(0, 0) << std::endl << std::endl;
	
	// class Eigen::Map represents only a matrix or vector EXPRESSION mapping an EXISTING array or data
	Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> firstChannel_(cv_eigen_container_[0].ptr<float>(), cv_eigen_.rows, cv_eigen_.cols);
	Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> secondChannel_(cv_eigen_container_[1].ptr<float>(), cv_eigen_.rows, cv_eigen_.cols);
	std::cout << "firstChannel_: " << std::endl << firstChannel_ << std::endl << std::endl;
	Eigen::ArrayXXf demo_array(3, 3);
	secondChannel_ = secondChannel_.array() + 999;
	std::cout << "secondChannel_: " << std::endl <<secondChannel_ << std::endl << std::endl;
	firstChannel_(0, 0) = 100;
	std::cout << "cv_eigen_container_[0]: " << cv_eigen_container_[0].at<float>(0, 0) << std::endl;

	


	// cv:filter2D testing

	float src_data[] = { 1, 1, 1, 2, 2, 2, 10, 10, 10};
	cv::Mat src(3, 3, CV_32FC1, src_data);
	float kernel_data[] = { -1, 0, 1 };
	cv::Mat kernel(3, 1, CV_32FC1, kernel_data);
	cv::Mat dst;
	cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);

	std::cout << "dst:" << std::endl << dst << std::endl;

	Eigen::Matrix3f src_eigen;
	src_eigen << 1, 1, 1,
				 2, 2, 2,
				 10, 10, 10;
	src_eigen = src_eigen.array() + src_eigen.array();
	std::cout << "src_eigen: " << std::endl << src_eigen << std::endl;

	Eigen::Vector3f kernel_eigen;
	kernel_eigen << -1, 0, 1;

	Eigen::Matrix3f dst_eigen;

	cv::Point anchor(kernel_eigen.cols() / 2, kernel_eigen.rows() / 2);
	for (int i = 0; i < src_eigen.rows(); i++)
	{
		for (int j = 0; j < src_eigen.cols(); j++)
		{
			int sum = 0;
			for (int k = 0; k < kernel_eigen.rows(); k++)
			{
				for (int l = 0; l < kernel_eigen.cols(); l++)
				{
					if ((i + k - anchor.y) < 0 || (j + l - anchor.x) < 0 || 
						(i + k - anchor.y) > (src_eigen.rows() - 1) || (j + l - anchor.x) > (src_eigen.cols() - 1))
						continue;
					sum += kernel_eigen(k, l) * src_eigen(i + k - anchor.y, j + l - anchor.x);
				}
			}
			dst_eigen(i, j) = sum;
		}
	}
	std::cout << "dst_eigen: " << std::endl << dst_eigen << std::endl;


	// complex number operations tesing
	Eigen::Matrix3f real, imag;
	real << 1, 3, 5,
			7, 9, 11,
			13, 15, 17;
	imag << 2, 4, 6,
			8, 10, 12,
			14, 16, 18;

	std::vector<Eigen::Matrix3f> complexa({ real, imag });
	
	return 0;
}
