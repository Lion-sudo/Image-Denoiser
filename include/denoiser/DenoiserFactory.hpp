#pragma once

#include <denoiser/ImageDenoiser.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace denoiser {

/// \brief Strongly-typed identifiers for the denoisers shipped with the library.
enum class DenoiserType {
    AnisotropicDiffusion,
    Bilateral,
    NonLocalMeans,
    Dct,
    TotalVariation,
};

/// \brief Registry-based factory for `ImageDenoiser` implementations.
///
/// The built-in denoisers are registered eagerly by the factory itself. This
/// deliberately avoids relying on TU-local static registrars: with static
/// libraries, the linker is free to drop translation units nothing references,
/// which would silently empty a self-registration-based registry.
///
/// Third-party code can still extend the registry at runtime through
/// \c registerDenoiser(); identifiers are matched case-insensitively.
///
/// Thread safety: the registry itself is the only mutable static state in the
/// library. Its initialization (including the five built-ins) is protected by
/// the C++ function-local-static guarantee, so \c create(), \c available() and
/// \c isRegistered() may be called concurrently from any number of threads
/// once initialization has happened. \c registerDenoiser() mutates the registry
/// and is therefore *not* synchronized: call it during startup, before
/// filtering threads exist. No lock is taken on the read paths, because that
/// would put a mutex on a path the concurrent runner calls per method.
class DenoiserFactory {
public:
    /// \brief A callable that produces a fresh, stateless denoiser instance.
    using Creator = std::function<std::unique_ptr<ImageDenoiser>()>;

    /// \brief Creates a built-in denoiser from its strongly-typed identifier.
    [[nodiscard]] static std::unique_ptr<ImageDenoiser> create(DenoiserType type);

    /// \brief Creates a denoiser by (case-insensitive) string identifier, e.g. "bilateral" or "nlm".
    /// \throws UnknownDenoiserError if the identifier is not registered; the
    ///         message lists every available identifier.
    [[nodiscard]] static std::unique_ptr<ImageDenoiser> create(const std::string& identifier);

    /// \brief Registers a custom denoiser under \p identifier.
    /// \returns false if the identifier (modulo case) is already taken.
    /// \throws InvalidParameterError if \p creator is not callable.
    static bool registerDenoiser(const std::string& identifier, Creator creator);

    /// \brief Convenience overload for stateless denoiser types.
    template <typename DenoiserT>
    static bool registerDenoiser(const std::string& identifier)
    {
        return registerDenoiser(identifier, [] { return std::make_unique<DenoiserT>(); });
    }

    /// \brief Sorted list of every registered identifier.
    [[nodiscard]] static std::vector<std::string> available();

    /// \brief True if \p identifier (modulo case) is registered.
    [[nodiscard]] static bool isRegistered(const std::string& identifier);

    /// \brief Canonical registry identifier of a built-in \p type.
    /// \throws UnknownDenoiserError for values outside the enum's valid range.
    [[nodiscard]] static std::string identifierFor(DenoiserType type);
};

} // namespace denoiser
