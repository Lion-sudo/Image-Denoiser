#pragma once

#include <denoiser/ImageDenoiser.hpp>

#include <vector>

namespace denoiser {

/// \brief Transform-domain denoiser: overlapping block DCT with soft-thresholding.
///
/// The [0, 1] float working buffer is split into channel planes. Each channel is
/// tiled with blockSize x blockSize blocks on a sliding grid (50% overlap,
/// step = blockSize / 2) to suppress blocking artifacts. Per block:
///
///   1. `cv::dct()` transforms the block into frequency coefficients.
///   2. Soft-thresholding shrinks every AC coefficient toward zero by tau:
///          c' = sign(c) * max(0, |c| - tau),  tau = threshold * intensitySigma
///      Noise lives mostly in small high-frequency coefficients and vanishes,
///      while strong structural coefficients survive (shrunk slightly).
///      The DC coefficient (index 0) is preserved so brightness is not biased.
///   3. `cv::idct()` reconstructs the block.
///
/// All overlapping reconstructions are averaged per pixel, which yields smooth
/// transitions at block boundaries; block origins are clamped at the image
/// borders so every pixel is covered by at least one block.
///
/// Block size: `DenoiseParams::blockSize` is clamped to fit the image and
/// rounded down to an even value, because `cv::dct()` implements only even
/// block sizes (blockSize 8 on a 5-pixel-wide image therefore uses 4x4
/// blocks). Images smaller than 2x2 have no valid block and are rejected with
/// an InvalidParameterError.
///
/// The shrinkage follows the wavelet-denoising tradition (Donoho & Johnstone's
/// universal shrinkage) and mirrors the DCT stage of BM3D (Dabov et al., 2007),
/// which uses the same transform with hard thresholds.
class DctFilter final : public ImageDenoiser {
public:
    DctFilter() = default;

    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params) const override;
    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                 const ProgressFn& onProgress) const override;

    [[nodiscard]] std::string name() const override { return "dct"; }

    [[nodiscard]] std::string description() const override
    {
        return "Frequency-domain filter: overlapping block DCT with soft-thresholded "
               "coefficients. Fast and effective against Gaussian noise.";
    }

private:
    /// \brief Validates DCT-specific parameters. \throws InvalidParameterError.
    static void validateParams(const DenoiseParams& params);

    /// \brief blockSize clamped to fit the image and rounded down to an even
    ///        value (`cv::dct()` implements only even block sizes).
    /// \pre The image is at least 2x2 pixels; checked by denoise().
    [[nodiscard]] static int effectiveBlockSize(const cv::Mat& working, const DenoiseParams& params);

    /// \brief Sign-preserving shrinkage of the AC coefficients: soft-threshold
    ///        with tau, leaving the DC coefficient (element 0) untouched.
    static void softThresholdInPlace(cv::Mat& coefficients, double threshold);

    /// \brief One filtering pass over the [0, 1] float working buffer.
    ///        \p rowProgress is invoked with the completed block-row fraction.
    [[nodiscard]] cv::Mat applyPass(const cv::Mat& working, const DenoiseParams& params,
                                    const ProgressFn& rowProgress) const;
};

} // namespace denoiser
