#pragma once

#include <denoiser/Error.hpp>

#include <opencv2/core.hpp>

#include <functional>
#include <string>

namespace denoiser {

/// \brief Hyperparameters shared by every denoising algorithm.
///
/// Intensity-based parameters (`intensitySigma`) are expressed in a normalized
/// [0, 1] scale: 8-bit and 16-bit inputs are converted to float in [0, 1]
/// internally, and the result is converted back to the original depth, so the
/// same parameter set behaves consistently across image types.
struct DenoiseParams {
    double spatialSigma = 3.0;    ///< Spatial Gaussian sigma (bilateral kernel radius grows with it).
    double intensitySigma = 0.1;  ///< Range sigma (bilateral) / h (NLM) / noise sigma (DCT), in [0, 1] intensity units.
    int searchWindow = 21;        ///< NLM search window side length; must be a positive odd integer.
    int patchSize = 7;            ///< NLM patch side length; must be a positive odd integer.
    int blockSize = 8;            ///< Transform-domain (DCT) block side length; larger blocks smooth more aggressively.
    double threshold = 2.7;       ///< Transform-domain soft-threshold multiplier: tau = threshold * intensitySigma.
    int num_iterations = 30;      ///< Explicit-scheme steps: PDE time steps (anisotropic) / projection steps (TV).
    double time_step = 0.15;      ///< Explicit step dt (anisotropic) / tau (TV); must lie in (0, 0.25] for stability.
    double conduction_coefficient = 0.15; ///< Perona-Malik k: gradients well above k are treated as edges.
    double lambda = 0.1;          ///< TV regularization weight (Chambolle): u = f - lambda * div(p); larger = smoother.
    int iterations = 1;           ///< Outer filtering passes (generic; the PDE has its own num_iterations time steps).

    /// \brief Validates the parameters used by every algorithm.
    /// \throws InvalidParameterError if any invariant is violated.
    void validate() const;

    /// \brief Additionally validates the patch-based parameters (NLM).
    /// \throws InvalidParameterError if `searchWindow` or `patchSize` are not positive odd integers.
    void validatePatchBased() const;

    /// \brief Additionally validates the transform-domain parameters (DCT).
    /// \throws InvalidParameterError if `blockSize` is outside [1, 64] or `threshold` is negative.
    ///         Note: `threshold = 0` is accepted and means "transform without shrinkage".
    void validateTransform() const;

    /// \brief Additionally validates the PDE parameters (anisotropic diffusion).
    /// \throws InvalidParameterError if `num_iterations` is outside [1, 500],
    ///         `time_step` is not in (0, 0.25], or `conduction_coefficient` is not strictly positive.
    void validateDiffusion() const;

    /// \brief Additionally validates the variational (TV) parameters.
    ///
    /// Chambolle's projection shares the step-count and stability bounds with
    /// the explicit PDE scheme, so validateDiffusion() is reused and only the
    /// regularization weight is checked here.
    /// \throws InvalidParameterError if `lambda` is not strictly positive.
    void validateVariational() const;
};

/// \brief Abstract interface every denoising algorithm must implement.
///
/// Implementations are stateless value objects: a single instance may be reused
/// for any number of images and parameter sets. All heavy work happens inside
/// \c denoise(); instances own no resources, so RAII is delegated to the
/// `std::unique_ptr<ImageDenoiser>` returned by the factory.
///
/// \c denoise() is \c const, so statelessness is enforced by the compiler
/// rather than by convention: an implementation cannot cache per-image state
/// in the object. Two threads may therefore call \c denoise() on the *same*
/// instance concurrently, as long as they pass distinct \p output matrices;
/// the input may be shared, because it is only ever read.
class ImageDenoiser {
public:
    ImageDenoiser() = default;
    virtual ~ImageDenoiser() = default;

    // Denoisers are created through the factory and owned via unique_ptr;
    // copying/moving polymorphic filters would slice the interface.
    ImageDenoiser(const ImageDenoiser&) = delete;
    ImageDenoiser& operator=(const ImageDenoiser&) = delete;
    ImageDenoiser(ImageDenoiser&&) = delete;
    ImageDenoiser& operator=(ImageDenoiser&&) = delete;

    /// \brief Denoises \p input into \p output.
    ///
    /// \p output is (re)allocated to match \p input's size, depth and channel
    /// count, so it may be an empty matrix or even alias \p input (the input is
    /// copied internally before any pixel is written).
    ///
    /// \throws InvalidParameterError for empty or unsupported images and invalid parameters.
    virtual void denoise(const cv::Mat& input, cv::Mat& output,
                         const DenoiseParams& params) const = 0;

    /// \brief Progress reporter: invoked with the completed fraction in [0, 1].
    ///
    /// Callbacks are invoked synchronously from inside \c denoise(); when the
    /// algorithm parallelizes internally (OpenMP), they may be invoked
    /// concurrently from several worker threads, so implementations must be
    /// thread-safe. Keep them cheap: they run inside the hot loops.
    ///
    /// A callback must not throw: it is called from OpenMP worker threads,
    /// where an escaping exception calls std::terminate instead of reaching
    /// the caller. Filters that call throwable APIs inside a parallel region
    /// (DctFilter) capture such failures and rethrow them on the calling
    /// thread, but a throwing progress callback is a caller error.
    using ProgressFn = std::function<void(double)>;

    /// \brief Denoises \p input into \p output, reporting progress.
    ///
    /// \p onProgress may be empty to skip reporting. The default implementation
    /// ignores it and runs to completion, so custom denoisers that only
    /// override the three-argument overload keep working unchanged.
    virtual void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                         const ProgressFn& onProgress) const
    {
        (void)onProgress;
        denoise(input, output, params);
    }

    /// \brief Canonical registry identifier, e.g. "bilateral".
    [[nodiscard]] virtual std::string name() const = 0;

    /// \brief Short human-readable description used by the CLI's `--list`.
    [[nodiscard]] virtual std::string description() const = 0;

protected:
    /// \brief Shared validation of the input image and generic parameters.
    /// \throws InvalidParameterError.
    static void validateCommon(const cv::Mat& input, const DenoiseParams& params);
};

} // namespace denoiser
