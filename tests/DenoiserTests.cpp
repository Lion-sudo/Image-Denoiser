// Self-contained test runner for the denoiser library.
//
// Deliberately framework-free: `ctest` works straight after a clone with
// nothing but OpenCV installed. Every check prints nothing on success, so a
// green run is quiet and a failing run names the exact contract that broke.
//
// Coverage focus:
//   - factory/registry behaviour and extensibility (OOP design),
//   - parameter and input validation (error handling),
//   - per-filter numeric sanity: geometry, type, finiteness, noise reduction,
//   - in-place aliasing (output aliases input) for every filter,
//   - depth/channel preservation,
//   - the progress-callback contract, including a regression test for
//     per-pixel instead of per-row reporting,
//   - statelessness: instance reuse and concurrent calls on one shared object.

#include <denoiser/DenoiserFactory.hpp>
#include <denoiser/Error.hpp>
#include <denoiser/ImageDenoiser.hpp>

#include <opencv2/core.hpp>

#include <atomic>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void section(const std::string& title)
{
    std::cout << "[" << title << "]\n";
}

void check(bool condition, const std::string& what)
{
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << "  FAIL: " << what << '\n';
    }
}

/// Asserts that \p fn throws \p Exception (and nothing else).
template <typename Exception, typename Fn>
void checkThrows(Fn fn, const std::string& what)
{
    ++g_checks;
    try {
        fn();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        ++g_failures;
        std::cerr << "  FAIL: " << what << " threw the wrong exception type: " << error.what()
                  << '\n';
        return;
    }
    ++g_failures;
    std::cerr << "  FAIL: " << what << " did not throw\n";
}

/// Test-only denoiser, used to prove the registry accepts third-party types.
class PassthroughFilter final : public denoiser::ImageDenoiser {
public:
    void denoise(const cv::Mat& input, cv::Mat& output,
                 const denoiser::DenoiseParams&) const override
    {
        output = input.clone();
    }

    [[nodiscard]] std::string name() const override { return "passthrough"; }

    [[nodiscard]] std::string description() const override
    {
        return "Test-only pass-through filter (returns the input unchanged).";
    }
};

constexpr double kNoiseSigma = 0.05;

/// A flat mid-gray image polluted with Gaussian noise: the ground truth is a
/// constant, so any reduction in standard deviation is genuine denoising.
cv::Mat makeNoisyImage(int rows, int cols, int channels)
{
    cv::Mat image(rows, cols, CV_MAKETYPE(CV_32F, channels), cv::Scalar::all(0.5));
    cv::Mat noise(image.size(), CV_MAKETYPE(CV_32F, channels));
    cv::randn(noise, 0.0, kNoiseSigma);
    image += noise;
    return image;
}

double stddevOf(const cv::Mat& image)
{
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(image.reshape(1), mean, stddev);
    return stddev[0];
}

bool identical(const cv::Mat& lhs, const cv::Mat& rhs)
{
    return cv::norm(lhs - rhs, cv::NORM_INF) == 0.0;
}

/// The five built-ins, listed explicitly so the suite does not silently change
/// meaning when a custom denoiser is registered mid-run.
const std::vector<std::string>& builtinIdentifiers()
{
    static const std::vector<std::string> identifiers = {"anisotropic", "bilateral", "dct",
                                                          "nlm", "tv"};
    return identifiers;
}

} // namespace

int main()
{
    using namespace denoiser;

    std::cout << "denoiser test suite\n";

    const DenoiseParams params; // documented defaults, valid for every filter
    params.validate();

    // -----------------------------------------------------------------------
    section("factory: create, identify, reject");
    // -----------------------------------------------------------------------
    for (const std::string& identifier : builtinIdentifiers()) {
        check(DenoiserFactory::isRegistered(identifier),
              "built-in '" + identifier + "' is registered");
        const std::unique_ptr<ImageDenoiser> denoiser = DenoiserFactory::create(identifier);
        check(denoiser != nullptr, "create('" + identifier + "') returns an instance");
        check(denoiser->name() == identifier,
              "name() of '" + identifier + "' matches its registry identifier");
        check(!denoiser->description().empty(),
              "description() of '" + identifier + "' is non-empty");
    }

    check(DenoiserFactory::isRegistered("BiLaTeRaL"),
          "identifiers are matched case-insensitively");
    check(!DenoiserFactory::isRegistered("no-such-filter"),
          "an unregistered identifier reports false");
    checkThrows<UnknownDenoiserError>(
        [] { (void)DenoiserFactory::create("no-such-filter"); },
        "create() with an unknown identifier");

    check(DenoiserFactory::identifierFor(DenoiserType::NonLocalMeans) == "nlm",
          "identifierFor(NonLocalMeans) == \"nlm\"");
    check(DenoiserFactory::create(DenoiserType::Dct)->name() == "dct",
          "enum-based create() and identifierFor() agree");
    checkThrows<UnknownDenoiserError>(
        [] { (void)DenoiserFactory::create(static_cast<DenoiserType>(999)); },
        "create() with an out-of-range enum value");

    // -----------------------------------------------------------------------
    section("validation: parameters");
    // -----------------------------------------------------------------------
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.searchWindow = 4; p.validatePatchBased(); },
        "even searchWindow is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.patchSize = 0; p.validatePatchBased(); },
        "zero patchSize is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.blockSize = 0; p.validateTransform(); },
        "zero blockSize is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.blockSize = 65; p.validateTransform(); },
        "blockSize above 64 is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.threshold = -1.0; p.validateTransform(); },
        "negative threshold is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.time_step = 0.5; p.validateDiffusion(); },
        "time_step above 0.25 (unstable) is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.time_step = 0.0; p.validateDiffusion(); },
        "zero time_step is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.num_iterations = 0; p.validateDiffusion(); },
        "zero num_iterations is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.conduction_coefficient = 0.0; p.validateDiffusion(); },
        "zero conduction_coefficient is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.lambda = 0.0; p.validateVariational(); },
        "zero lambda is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.spatialSigma = 0.0; p.validate(); },
        "zero spatialSigma is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.intensitySigma = -0.1; p.validate(); },
        "negative intensitySigma is rejected");
    checkThrows<InvalidParameterError>(
        [] { DenoiseParams p; p.iterations = 0; p.validate(); },
        "zero iterations is rejected");

    // -----------------------------------------------------------------------
    section("validation: input images");
    // -----------------------------------------------------------------------
    {
        const cv::Mat empty;
        for (const std::string& identifier : builtinIdentifiers()) {
            const auto denoiser = DenoiserFactory::create(identifier);
            cv::Mat output;
            checkThrows<InvalidParameterError>(
                [&] { denoiser->denoise(empty, output, params); },
                identifier + ": empty input is rejected");
        }

        cv::Mat wrongDepth;
        makeNoisyImage(8, 8, 1).convertTo(wrongDepth, CV_64F);
        const auto denoiser = DenoiserFactory::create("bilateral");
        cv::Mat output;
        checkThrows<InvalidParameterError>(
            [&] { denoiser->denoise(wrongDepth, output, params); },
            "unsupported depth (CV_64F) is rejected");

        const cv::Mat noisy = makeNoisyImage(8, 8, 1);
        DenoiseParams badParams;
        badParams.spatialSigma = 0.0;
        checkThrows<InvalidParameterError>(
            [&] { denoiser->denoise(noisy, output, badParams); },
            "invalid parameters are rejected before filtering");
    }

    // -----------------------------------------------------------------------
    section("filters: geometry, type, finiteness, noise reduction");
    // -----------------------------------------------------------------------
    const cv::Mat noisy = makeNoisyImage(32, 32, 3);
    const double noisyStddev = stddevOf(noisy);
    check(noisyStddev > 0.02, "the synthetic test image actually contains noise");

    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "  noisy input stddev: " << noisyStddev << '\n';

    for (const std::string& identifier : builtinIdentifiers()) {
        const auto denoiser = DenoiserFactory::create(identifier);

        cv::Mat output;
        denoiser->denoise(noisy, output, params);

        check(output.size() == noisy.size(), identifier + ": preserves image geometry");
        check(output.type() == noisy.type(), identifier + ": preserves depth and channels");
        check(cv::checkRange(output, true), identifier + ": output contains no NaN or Inf");
        check(output.data != noisy.data, identifier + ": writes into its own buffer");

        const double outputStddev = stddevOf(output);
        std::cout << "  " << identifier << ": stddev " << outputStddev << " ("
                  << (100.0 * (1.0 - outputStddev / noisyStddev)) << "% reduction)\n";
        check(outputStddev < 0.8 * noisyStddev, identifier + ": reduces noise");

        // Output aliasing the input must be safe: the filter copies the input
        // into a private buffer before writing any pixel.
        cv::Mat inPlace = noisy.clone();
        denoiser->denoise(inPlace, inPlace, params);
        check(identical(inPlace, output),
              identifier + ": in-place call (output aliases input) matches out-of-place");
    }
    std::cout.unsetf(std::ios::fixed);

    // -----------------------------------------------------------------------
    section("filters: depth and channel handling");
    // -----------------------------------------------------------------------
    for (const int depth : {CV_8U, CV_16U, CV_32F}) {
        const double scale = depth == CV_8U ? 255.0 : depth == CV_16U ? 65535.0 : 1.0;
        cv::Mat image;
        noisy.convertTo(image, depth, scale);

        const auto denoiser = DenoiserFactory::create("bilateral");
        cv::Mat output;
        denoiser->denoise(image, output, params);
        check(output.depth() == depth, "bilateral preserves input depth " + std::to_string(depth));
        check(output.channels() == image.channels(), "bilateral preserves channel count");
    }

    for (const int channels : {1, 4}) {
        const cv::Mat image = makeNoisyImage(16, 16, channels);
        const auto denoiser = DenoiserFactory::create("nlm");
        cv::Mat output;
        denoiser->denoise(image, output, params);
        check(output.channels() == channels,
              "nlm handles " + std::to_string(channels) + " channel(s)");
    }

    // -----------------------------------------------------------------------
    section("filters: degenerate geometry (border and padding stress)");
    // -----------------------------------------------------------------------
    // Images smaller than the kernel radius, the search window and the DCT
    // block are where padding/border math goes wrong, so every filter must
    // survive 1-pixel-tall/wide and sub-block images.
    //
    // Contract: a degenerate image is either processed correctly, or rejected
    // with a catchable InvalidParameterError. Any other exception - and above
    // all std::terminate from an exception escaping an OpenMP region - is a
    // failure. Ordinary sized images must always be processed.
    for (const std::string& identifier : builtinIdentifiers()) {
        const auto denoiser = DenoiserFactory::create(identifier);
        for (const cv::Size size : {cv::Size(1, 1), cv::Size(2, 2), cv::Size(3, 5),
                                    cv::Size(5, 3), cv::Size(1, 17), cv::Size(17, 1)}) {
            const cv::Mat tiny = makeNoisyImage(size.height, size.width, 3);
            const std::string label = identifier + " on " + std::to_string(size.width) + "x"
                                      + std::to_string(size.height);
            cv::Mat output;
            bool rejected = false;
            bool unexpected = false;
            std::string message;
            try {
                denoiser->denoise(tiny, output, params);
            } catch (const InvalidParameterError& error) {
                rejected = true;
                message = error.what();
            } catch (const std::exception& error) {
                unexpected = true;
                message = error.what();
            }

            check(!unexpected, label + ": fails only with a documented error (" + message + ")");
            if (size.width >= 2 && size.height >= 2) {
                check(!rejected, label + ": is processed, not rejected (" + message + ")");
            }
            if (!rejected && !unexpected) {
                check(output.size() == size, label + ": preserves geometry");
                check(cv::checkRange(output, true), label + ": output is finite");
            }
        }
    }

    // -----------------------------------------------------------------------
    section("progress callback contract");
    // -----------------------------------------------------------------------
    for (const std::string& identifier : builtinIdentifiers()) {
        const auto denoiser = DenoiserFactory::create(identifier);
        cv::Mat output;
        std::atomic<int> calls{0};
        std::atomic<bool> inRange{true};
        std::atomic<double> lastReported{0.0};

        denoiser->denoise(noisy, output, params,
                          [&calls, &inRange, &lastReported](double fraction) {
                              calls.fetch_add(1, std::memory_order_relaxed);
                              if (!(fraction >= 0.0 && fraction <= 1.0)) {
                                  inRange.store(false, std::memory_order_relaxed);
                              }
                              lastReported.store(fraction, std::memory_order_relaxed);
                          });

        check(calls.load() > 0, identifier + ": reports progress at least once");
        check(inRange.load(), identifier + ": every reported fraction is within [0, 1]");
        check(lastReported.load() == 1.0, identifier + ": finishes at 100%");
    }

    {
        // Regression: progress used to be reported once per *pixel* instead of
        // once per row, which both saturated the fraction inside the first row
        // and put an atomic read-modify-write in the innermost loop.
        const auto bilateral = DenoiserFactory::create("bilateral");
        std::atomic<int> calls{0};
        cv::Mat output;
        bilateral->denoise(noisy, output, params, [&calls](double) {
            calls.fetch_add(1, std::memory_order_relaxed);
        });
        check(calls.load() <= noisy.rows + 2,
              "bilateral reports progress per row, not per pixel (got "
                  + std::to_string(calls.load()) + " callbacks for "
                  + std::to_string(noisy.rows) + " rows)");
    }

    // -----------------------------------------------------------------------
    section("statelessness and thread safety");
    // -----------------------------------------------------------------------
    {
        const auto denominator = DenoiserFactory::create("bilateral");
        cv::Mat first;
        cv::Mat second;
        denominator->denoise(noisy, first, params);
        denominator->denoise(noisy, second, params);
        check(identical(first, second),
              "reusing one instance for a second run gives identical results");
    }

    {
        // One shared const instance, one shared read-only input, six concurrent
        // calls with independent outputs. Because denoise() is const and
        // stateless, the results must be bit-identical to the serial run.
        const std::unique_ptr<ImageDenoiser> shared = DenoiserFactory::create("bilateral");
        cv::Mat reference;
        shared->denoise(noisy, reference, params);

        std::vector<std::future<cv::Mat>> futures;
        futures.reserve(6);
        for (int i = 0; i < 6; ++i) {
            futures.push_back(std::async(std::launch::async, [&shared, &noisy, &params] {
                cv::Mat output;
                shared->denoise(noisy, output, params);
                return output;
            }));
        }

        bool allIdentical = true;
        for (std::future<cv::Mat>& future : futures) {
            allIdentical = allIdentical && identical(future.get(), reference);
        }
        check(allIdentical,
              "one shared instance across 6 threads matches the serial result");
    }

    // -----------------------------------------------------------------------
    section("registry extensibility");
    // -----------------------------------------------------------------------
    // Registered last: this mutates the process-wide registry, so everything
    // above works against exactly the five built-ins.
    check(DenoiserFactory::registerDenoiser<PassthroughFilter>("passthrough"),
          "a third-party denoiser can be registered");
    check(!DenoiserFactory::registerDenoiser<PassthroughFilter>("PASSTHROUGH"),
          "duplicate registration is rejected, case-insensitively");
    checkThrows<InvalidParameterError>(
        [] { (void)DenoiserFactory::registerDenoiser("broken", DenoiserFactory::Creator{}); },
        "an empty creator is rejected");

    {
        const auto custom = DenoiserFactory::create("passthrough");
        check(custom->name() == "passthrough", "a custom denoiser reports its own name");

        // A custom filter that only implements the three-argument overload must
        // still work when called through the progress-reporting overload: the
        // base class forwards and simply skips reporting.
        cv::Mat output;
        int progressCalls = 0;
        custom->denoise(noisy, output, params, [&progressCalls](double) { ++progressCalls; });
        check(identical(output, noisy), "a custom denoiser runs through the interface");
        check(progressCalls == 0,
              "the default progress overload forwards and reports nothing (documented)");
    }

    check(DenoiserFactory::available().size() == builtinIdentifiers().size() + 1,
          "the registry lists the built-ins plus the custom denoiser");

    // -----------------------------------------------------------------------
    std::cout << '\n'
              << (g_failures == 0 ? "PASSED" : "FAILED") << ": " << (g_checks - g_failures)
              << '/' << g_checks << " checks\n";
    return g_failures == 0 ? 0 : 1;
}
