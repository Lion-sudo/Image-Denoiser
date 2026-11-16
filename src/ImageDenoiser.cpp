#include <denoiser/ImageDenoiser.hpp>

#include <string>

namespace denoiser {

void DenoiseParams::validate() const
{
    if (!(spatialSigma > 0.0)) {
        throw InvalidParameterError("spatialSigma must be strictly positive (got "
                                    + std::to_string(spatialSigma) + ")");
    }
    if (!(intensitySigma > 0.0)) {
        throw InvalidParameterError("intensitySigma must be strictly positive (got "
                                    + std::to_string(intensitySigma) + ")");
    }
    if (iterations < 1) {
        throw InvalidParameterError("iterations must be at least 1 (got "
                                    + std::to_string(iterations) + ")");
    }
}

void DenoiseParams::validatePatchBased() const
{
    validate();
    if (searchWindow < 1 || (searchWindow % 2) == 0) {
        throw InvalidParameterError("searchWindow must be a positive odd integer (got "
                                    + std::to_string(searchWindow) + ")");
    }
    if (patchSize < 1 || (patchSize % 2) == 0) {
        throw InvalidParameterError("patchSize must be a positive odd integer (got "
                                    + std::to_string(patchSize) + ")");
    }
}

void DenoiseParams::validateTransform() const
{
    validate();
    if (blockSize < 1 || blockSize > 64) {
        throw InvalidParameterError("blockSize must be within [1, 64] (got "
                                    + std::to_string(blockSize) + ")");
    }
    if (!(threshold >= 0.0)) {
        throw InvalidParameterError("threshold must be non-negative (got "
                                    + std::to_string(threshold) + ")");
    }
}

void DenoiseParams::validateDiffusion() const
{
    validate();
    if (num_iterations < 1 || num_iterations > 500) {
        throw InvalidParameterError("num_iterations must be within [1, 500] (got "
                                    + std::to_string(num_iterations) + ")");
    }
    if (!(time_step > 0.0) || time_step > 0.25) {
        throw InvalidParameterError("time_step must be in (0, 0.25] for numerical stability (got "
                                    + std::to_string(time_step) + ")");
    }
    if (!(conduction_coefficient > 0.0)) {
        throw InvalidParameterError("conduction_coefficient must be strictly positive (got "
                                    + std::to_string(conduction_coefficient) + ")");
    }
}

void DenoiseParams::validateVariational() const
{
    // num_iterations and time_step bounds are shared with the PDE scheme.
    validateDiffusion();
    if (!(lambda > 0.0)) {
        throw InvalidParameterError("lambda must be strictly positive (got "
                                    + std::to_string(lambda) + ")");
    }
}

void ImageDenoiser::validateCommon(const cv::Mat& input, const DenoiseParams& params)
{
    if (input.empty()) {
        throw InvalidParameterError("input image is empty");
    }

    const int depth = input.depth();
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F) {
        throw InvalidParameterError("unsupported input depth " + std::to_string(depth)
                                    + " (expected CV_8U, CV_16U or CV_32F)");
    }

    const int channels = input.channels();
    if (channels < 1 || channels > 4) {
        throw InvalidParameterError("unsupported channel count "
                                    + std::to_string(channels) + " (expected 1, 2, 3 or 4)");
    }

    params.validate();
}

} // namespace denoiser
