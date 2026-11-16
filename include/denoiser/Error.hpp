#pragma once

#include <stdexcept>
#include <string>

namespace denoiser {

/// \brief Base type for every error raised by the denoiser library.
///
/// Catching `DenoiserError` is sufficient to handle all library-level failures.
class DenoiserError : public std::runtime_error {
public:
    explicit DenoiserError(const std::string& message)
        : std::runtime_error(message) {}
};

/// \brief Raised when caller-supplied parameters or images violate the library contracts.
class InvalidParameterError final : public DenoiserError {
public:
    explicit InvalidParameterError(const std::string& message)
        : DenoiserError(message) {}
};

/// \brief Raised when the factory is asked for an unregistered denoiser identifier.
class UnknownDenoiserError final : public DenoiserError {
public:
    explicit UnknownDenoiserError(const std::string& message)
        : DenoiserError(message) {}
};

} // namespace denoiser
