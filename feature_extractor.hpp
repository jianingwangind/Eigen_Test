#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <thread>

#include "parameters.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "debug.hpp"

#include "vfc/image/vfc_image.hpp"
#include "vfc/core/vfc_core_all.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef USE_SIMD
#include "gradient.hpp"
#endif

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/caffe.hpp>
#endif

namespace eco
{
class  FeatureExtractor
{
  public:
	FeatureExtractor() {}
	virtual ~FeatureExtractor(){};

	ECO_FEATS extractor(const cv::Mat image,
						const cv::Point2f pos,
						const vector<float> scales,
						const EcoParameters &params,
						const bool &is_color_image);

	cv::Mat sample_patch(const cv::Mat im,
						 const cv::Point2f pos,
						 cv::Size2f sample_sz,
						 cv::Size2f input_sz);

#ifdef USE_SIMD
	vector<cv::Mat> get_hog_features_simd(const vector<cv::Mat> ims);
#else
	vector<cv::Mat> get_hog_features(const vector<cv::Mat> ims);
#endif
	vector<cv::Mat> hog_feature_normalization(vector<cv::Mat> &hog_feat_maps);
	inline vector<cv::Mat> get_hog_feats() const { return hog_feat_maps_; }

	vector<cv::Mat> get_cn_features(const vector<cv::Mat> ims);
	vector<cv::Mat> cn_feature_normalization(vector<cv::Mat> &cn_feat_maps);
	inline vector<cv::Mat> get_cn_feats() const { return cn_feat_maps_; }

#ifdef USE_CAFFE
	ECO_FEATS get_cnn_layers(vector<cv::Mat> im, const cv::Mat &deep_mean_mat);
	cv::Mat sample_pool(const cv::Mat &im, int smaple_factor, int stride);
	void cnn_feature_normalization(ECO_FEATS &feature);
	inline ECO_FEATS get_cnn_feats() const { return cnn_feat_maps_; }
#endif

  private:
	EcoParameters params_;

	HogFeatures hog_features_;
	int hog_feat_ind_ = -1;
	vector<cv::Mat> hog_feat_maps_;

	ColorspaceFeatures colorspace_features_;
	int colorspace_feat_ind_ = -1;
	vector<cv::Mat> colorspace_feat_maps_;

	CnFeatures cn_features_;
	int cn_feat_ind_ = -1;
	vector<cv::Mat> cn_feat_maps_;

	IcFeatures ic_features_;
	int ic_feat_ind_ = -1;
	vector<cv::Mat> ic_feat_maps_;


#ifdef USE_CAFFE
	boost::shared_ptr<caffe::Net<float>> net_;
	CnnFeatures cnn_features_;
	int cnn_feat_ind_ = -1;
	ECO_FEATS cnn_feat_maps_;
#endif
};

struct HoGHistogram9 {
	vfc::int32_t cnt[9];
	void clear() { for (vfc::int32_t i = 0; i < 9; i++) cnt[i] = 0; }
	void add(HoGHistogram9& b) { for (vfc::int32_t i = 0; i < 9; i++) cnt[i] += b.cnt[i]; }
};

class Fast_HoGFeature {
public:
	Fast_HoGFeature() {
		normalized_ = true;
		variablePartition_ = false;
		cell_size_ = 6;
		features_dim_ = 9;
	}
	
	void computeCell(vfc::float32_t* vec36,
		vfc::int32_t left, vfc::int32_t top,
		vfc::int32_t w, vfc::int32_t h);
	
	void compute(vfc::float32_t* vec36, bool normalize = true);

	void Fast_HoGFeature::compute(vfc::float32_t* vec36, vfc::int32_t cell_size, bool normalize = true);

	std::vector<cv::Mat> computeHOGFeature(const std::vector<cv::Mat> ims);



private:
	vfc::TImage<HoGHistogram9> integralImage_;
	bool variablePartition_;
	bool normalized_;
	vfc::int32_t cell_size_;
	vfc::int32_t sizeX_;
	vfc::int32_t sizeY_;
	vfc::int32_t features_dim_;
};

void computeGradientMagnitudes(vfc::int32_t f_startY, vfc::int32_t f_endY,
	vfc::int32_t f_endX,
	const cv::Mat& f_inputImg,
	vfc::TImageView<HoGHistogram9>& f_integralImg);

vfc::int32_t getBinIdx(vfc::uint8_t f_dx, vfc::uint8_t f_dy);

void integrateGradientX(vfc::int32_t f_startY, vfc::int32_t f_endY,
	vfc::int32_t f_startX, vfc::int32_t f_endX,
	vfc::TImageView<HoGHistogram9>& f_integralImg);

void integrateGradientY(vfc::int32_t f_startX, vfc::int32_t f_endX,
	vfc::int32_t f_startY, vfc::int32_t f_endY,
	vfc::TImageView<HoGHistogram9>& f_integralImg);

void computeIntegralHistogram(const cv::Mat& img, vfc::TImage<HoGHistogram9>& integral);

} // namespace eco
#endif
