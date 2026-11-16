#pragma once

#include <denoiser/ImageDenoiser.hpp>

namespace denoiser {

/// \brief Non-local means denoiser (Buades, Coll & Morel, CVPR 2005).
///
/// Each pixel is replaced by the weighted average of every pixel in its search
/// window, where the weight decays exponentially with the squared Euclidean
/// distance between the surrounding patches:
///
///     w(p, q) = exp(-mean_{i in Patch(p), c} (I_c(p + i) - I_c(q + i))^2 / h^2)
///
/// The distance is averaged over patch area and channels so that
/// `DenoiseParams::intensitySigma` (h) keeps a consistent meaning regardless of
/// patch size or channel count. Patch distances are computed on a padded
/// [0, 1] float buffer (BORDER_REFLECT_101), which removes all boundary checks.
class NonLocalMeansFilter final : public ImageDenoiser {
public:
    NonLocalMeansFilter() = default;

    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params) const override;
    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                 const ProgressFn& onProgress) const override;

    [[nodiscard]] std::string name() const override { return "nlm"; }

    [[nodiscard]] std::string description() const override
    {
        return "Patch-based self-similarity filter averaging pixels by patch "
               "similarity (Buades, Coll & Morel, 2005). Computationally intensive.";
    }

private:
    /// \brief Validates NLM-specific parameters (odd window/patch sizes). \throws InvalidParameterError.
    static void validateParams(const DenoiseParams& params);

    /// \brief One filtering pass over the [0, 1] float working buffer.
    ///        \p rowProgress is invoked with the completed row fraction.
    [[nodiscard]] cv::Mat applyPass(const cv::Mat& working, const DenoiseParams& params,
                                    const ProgressFn& rowProgress) const;

    /// \brief Mean squared difference between the patches anchored at (x1, y1) and
    ///        (x2, y2) inside \p padded, averaged over patch area and channels.
    ///
    /// All anchor coordinates are given in the padded coordinate system and must
    /// keep the full patch inside the padded image.
    [[nodiscard]] static double patchSquaredDistance(const cv::Mat& padded, int x1, int y1,
                                                     int x2, int y2, int halfPatch, int channels);

    /// \brief Converts a patch distance into a blending weight: exp(-distance / h^2).
    [[nodiscard]] static double weightFromDistance(double distance, double intensitySigma);
};

} // namespace denoiser
