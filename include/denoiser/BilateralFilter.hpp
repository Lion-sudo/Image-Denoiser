#pragma once

#include <denoiser/ImageDenoiser.hpp>

#include <vector>

namespace denoiser {

/// \brief Edge-preserving bilateral filter (Tomasi & Manduchi, ICCV 1998).
///
/// Each output pixel is a Gaussian-weighted average of its neighborhood, where
/// the weight combines a spatial Gaussian (exp of squared pixel distance,
/// parameterized by `DenoiseParams::spatialSigma`) with a per-channel
/// intensity-range Gaussian (parameterized by `DenoiseParams::intensitySigma`).
/// Per-channel range weights avoid color bleeding across edges where only one
/// channel changes. Work happens on a padded [0, 1] float buffer, so borders
/// need no special cases.
class BilateralFilter final : public ImageDenoiser {
public:
    BilateralFilter() = default;

    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params) const override;
    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                 const ProgressFn& onProgress) const override;

    [[nodiscard]] std::string name() const override { return "bilateral"; }

    [[nodiscard]] std::string description() const override
    {
        return "Edge-preserving Gaussian filter combining spatial closeness with "
               "intensity similarity (Tomasi & Manduchi, 1998).";
    }

private:
    /// \brief Precomputed spatial Gaussian look-up table for all offsets in the kernel.
    struct SpatialKernel {
        int radius = 0;               ///< Kernel radius: ceil(2 * spatialSigma).
        std::vector<float> weights;   ///< Row-major, index (dy + radius) * diameter + (dx + radius).
    };

    /// \brief Validates bilateral-specific parameter bounds. \throws InvalidParameterError.
    static void validateParams(const DenoiseParams& params);

    /// \brief Builds exp(-|offset|^2 / 2 sigma_s^2) for every offset in the kernel.
    [[nodiscard]] SpatialKernel buildSpatialKernel(double spatialSigma) const;

    /// \brief One filtering pass over the padded [0, 1] float buffer.
    ///        \p rowProgress is invoked with the completed row fraction.
    [[nodiscard]] cv::Mat applyPass(const cv::Mat& working, const SpatialKernel& kernel,
                                    double intensitySigma, const ProgressFn& rowProgress) const;
};

} // namespace denoiser
