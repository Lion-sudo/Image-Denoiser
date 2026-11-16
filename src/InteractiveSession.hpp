#pragma once

// Internal interactive (menu-driven) front-end for the denoiser CLI.
// Not part of the public library API.

namespace denoiser::cli {

/// \brief Runs the menu-driven interactive session.
///
/// Flow: ask for an image path, optionally pollute it with synthetic noise,
/// present the registered denoisers, tune parameters (Enter accepts defaults),
/// run the filter and save the result, then offer "run again / new image /
/// quit". Reads from stdin; closed or exhausted input ends the session
/// gracefully instead of spinning.
///
/// \returns the process exit code (always 0; input errors are re-prompted).
int runInteractive();

} // namespace denoiser::cli
