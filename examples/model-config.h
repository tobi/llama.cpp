// Modelfile parsing, layered config merge, and hardware-aware overrides.
//
// A Modelfile is a line-based text file declaring a model and its runtime
// parameters. Tokens are case-insensitive and recognised at the start of a
// line. Lines beginning with '#' are comments. Triple-quoted blocks are
// supported for SYSTEM and TEMPLATE values that span multiple lines.
//
//   NAME              <alias>
//   FROM              <path-to-gguf>
//   TEMPLATE          <single-line>     | TEMPLATE """ ... """
//   SYSTEM            <single-line>     | SYSTEM """ ... """
//   PARAMETER         <key> <value>
//   TAG               <tag>
//   SPECULATIVE_MODEL <path>
//   SPECULATIVE_MAX   <int>
//   SPECULATIVE_MIN   <int>
//   HARDWARE          <tier> <key> <value>
//
// Tier names are of the form vram_<low>_<high> (in gigabytes, inclusive low,
// exclusive high). The matcher applies the single tier whose range covers the
// detected VRAM. RAM-only systems use vram_0_<high> tiers with low GiB caps.

#pragma once

#include "common.h"

#include <string>
#include <vector>
#include <map>

struct hw_info {
    int  vram_gb     = 0;     // primary GPU VRAM, 0 = none detected
    int  ram_gb      = 0;     // system RAM
    std::string gpu_name;     // e.g. "NVIDIA GeForce RTX 4090" or "" if CPU
};

struct speculative_def {
    std::string model;        // draft model path
    int         max_draft = 0;
    int         min_draft = 0;
};

struct hw_tier {
    int low_gb  = 0;
    int high_gb = 0;
    std::map<std::string, std::string> params;  // raw key -> value, applied through apply_param
};

struct model_def {
    std::string name;                 // alias, becomes params.model_alias
    std::string source_path;          // path to the .modelfile this came from
    std::string template_text;        // chat template (currently informational)
    std::string system_text;          // default system prompt
    std::vector<std::string> tags;
    speculative_def speculative;

    gpt_params  params;               // populated from FROM + PARAMETER + HARDWARE
    std::vector<hw_tier> hw_tiers;    // unapplied tiers (kept for introspection)
};

// Detect the primary GPU's VRAM via nvidia-smi, and system RAM via sysconf.
// Returns zero-filled info on failure; never throws.
hw_info hw_detect();

// Parse a Modelfile. Returns true and populates `out`; on failure returns
// false and writes a diagnostic to `err`. The caller may pass an existing
// `defaults` gpt_params whose values are used where the Modelfile is silent.
bool model_def_load(const std::string & path,
                    const gpt_params & defaults,
                    model_def & out,
                    std::string & err);

// Apply the highest-matching hardware tier (greedy: pick the tier whose
// [low_gb, high_gb) range contains hw.vram_gb; if multiple, pick widest high).
// No-op if no tier matches.
void model_def_apply_hw(model_def & def, const hw_info & hw);

// Apply a single PARAMETER key/value onto gpt_params. Returns false for
// unknown keys (caller decides whether that is fatal). The key set mirrors
// the long-form names of gpt_params_parse, minus leading dashes.
bool apply_param(gpt_params & params, const std::string & key, const std::string & value);

// Helpers for the manager: scan a directory for *.modelfile files.
std::vector<std::string> scan_modelfiles(const std::string & dir);

// Trim whitespace in place.
std::string trim(const std::string & s);

// Render a small JSON object describing a model_def (alias, path, tags,
// resolved params). Useful for /slots, /v1/models endpoints. Returns a
// pretty single-line JSON string with no external dependency.
std::string model_def_to_json(const model_def & def);
