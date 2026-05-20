#include "model-config.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
    return s.substr(a, b - a);
}

static std::string lower(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
        0 == s.compare(s.size() - suffix.size(), suffix.size(), suffix);
}

static bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() &&
        0 == s.compare(0, prefix.size(), prefix);
}

// Split a string on whitespace into at most `max_parts` parts; the last part
// is the unconsumed tail (preserving inner whitespace and quotes).
static std::vector<std::string> split_n(const std::string & s, size_t max_parts) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size() && out.size() + 1 < max_parts) {
        while (i < s.size() && std::isspace((unsigned char) s[i])) ++i;
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && !std::isspace((unsigned char) s[j])) ++j;
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    while (i < s.size() && std::isspace((unsigned char) s[i])) ++i;
    if (i < s.size()) {
        out.push_back(s.substr(i));
    }
    return out;
}

bool apply_param(gpt_params & p, const std::string & key_in, const std::string & value) {
    std::string key = lower(key_in);
    // accept both - and _ as separators
    std::replace(key.begin(), key.end(), '-', '_');

    auto as_int = [&]() { return std::stoi(value); };
    auto as_float = [&]() { return std::stof(value); };
    auto as_bool = [&]() {
        std::string v = lower(value);
        return v == "true" || v == "1" || v == "yes" || v == "on";
    };

    if      (key == "seed")              p.seed = as_int();
    else if (key == "n_threads" ||
             key == "threads")           p.n_threads = as_int();
    else if (key == "n_predict")         p.n_predict = as_int();
    else if (key == "n_ctx" ||
             key == "ctx_size")          p.n_ctx = as_int();
    else if (key == "n_batch" ||
             key == "batch_size")        p.n_batch = std::min(512, as_int());
    else if (key == "n_keep" ||
             key == "keep")              p.n_keep = as_int();
    else if (key == "n_gpu_layers" ||
             key == "gpu_layers" ||
             key == "ngl")               p.n_gpu_layers = as_int();
    else if (key == "main_gpu")          p.main_gpu = as_int();
    else if (key == "low_vram")          p.low_vram = as_bool();
    else if (key == "top_k")             p.top_k = as_int();
    else if (key == "top_p")             p.top_p = as_float();
    else if (key == "tfs" ||
             key == "tfs_z")             p.tfs_z = as_float();
    else if (key == "typical" ||
             key == "typical_p")         p.typical_p = as_float();
    else if (key == "temp" ||
             key == "temperature")       p.temp = as_float();
    else if (key == "repeat_penalty")    p.repeat_penalty = as_float();
    else if (key == "repeat_last_n")     p.repeat_last_n = as_int();
    else if (key == "frequency_penalty") p.frequency_penalty = as_float();
    else if (key == "presence_penalty")  p.presence_penalty = as_float();
    else if (key == "mirostat")          p.mirostat = as_int();
    else if (key == "mirostat_tau")      p.mirostat_tau = as_float();
    else if (key == "mirostat_eta" ||
             key == "mirostat_lr")       p.mirostat_eta = as_float();
    else if (key == "memory_f16")        p.memory_f16 = as_bool();
    else if (key == "use_mmap")          p.use_mmap = as_bool();
    else if (key == "use_mlock" ||
             key == "mlock")             p.use_mlock = as_bool();
    else if (key == "embedding")         p.embedding = as_bool();
    else if (key == "penalize_nl")       p.penalize_nl = as_bool();
    else if (key == "lora")              { p.lora_adapter = value; p.use_mmap = false; }
    else if (key == "lora_base")         p.lora_base = value;
    else if (key == "tensor_split") {
        // comma- or slash-separated floats
        std::string buf;
        size_t idx = 0;
        for (size_t i = 0; i <= value.size() && idx < LLAMA_MAX_DEVICES; ++i) {
            char c = i < value.size() ? value[i] : ',';
            if (c == ',' || c == '/') {
                if (!buf.empty()) {
                    p.tensor_split[idx++] = std::stof(buf);
                    buf.clear();
                }
            } else if (!std::isspace((unsigned char) c)) {
                buf.push_back(c);
            }
        }
        for (; idx < LLAMA_MAX_DEVICES; ++idx) p.tensor_split[idx] = 0.0f;
    }
    else return false;

    return true;
}

// Read the next token starting at `pos`, skipping whitespace. Returns "" at EOF.
static std::string read_word(const std::string & s, size_t & pos) {
    while (pos < s.size() && std::isspace((unsigned char) s[pos])) ++pos;
    size_t start = pos;
    while (pos < s.size() && !std::isspace((unsigned char) s[pos])) ++pos;
    return s.substr(start, pos - start);
}

static bool parse_hw_tier_name(const std::string & name, int & low, int & high) {
    // expect: vram_<int>_<int>
    if (!starts_with(name, "vram_")) return false;
    size_t p1 = 5;
    size_t p2 = name.find('_', p1);
    if (p2 == std::string::npos) return false;
    try {
        low  = std::stoi(name.substr(p1, p2 - p1));
        high = std::stoi(name.substr(p2 + 1));
    } catch (...) { return false; }
    return true;
}

bool model_def_load(const std::string & path,
                    const gpt_params & defaults,
                    model_def & out,
                    std::string & err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open Modelfile: " + path;
        return false;
    }

    out.params = defaults;
    out.source_path = path;
    bool saw_from = false;

    std::vector<std::string> lines;
    {
        std::string line;
        while (std::getline(f, line)) lines.push_back(line);
    }

    // Helper to consume a value that may be a triple-quoted block. Starts
    // with the rest-of-line tail after a keyword; if that tail begins with
    // """ the block continues until a matching """ (on this or a later line).
    auto consume_value = [&](size_t & i, const std::string & first_tail) -> std::string {
        std::string tail = first_tail;
        // Trim leading whitespace from tail
        size_t a = 0;
        while (a < tail.size() && std::isspace((unsigned char) tail[a])) ++a;
        tail.erase(0, a);

        if (!starts_with(tail, "\"\"\"")) {
            return trim(tail);
        }
        tail.erase(0, 3);
        std::string acc;
        // Check if closing """ is on the same line
        size_t close = tail.find("\"\"\"");
        if (close != std::string::npos) {
            return tail.substr(0, close);
        }
        acc = tail;
        while (++i < lines.size()) {
            const std::string & ln = lines[i];
            size_t c = ln.find("\"\"\"");
            if (c != std::string::npos) {
                if (!acc.empty()) acc += "\n";
                acc += ln.substr(0, c);
                return acc;
            }
            if (!acc.empty()) acc += "\n";
            acc += ln;
        }
        // Unterminated block — treat what we have as value
        return acc;
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string raw = lines[i];
        const std::string t   = trim(raw);
        if (t.empty() || t[0] == '#') continue;

        size_t pos = 0;
        std::string kw = lower(read_word(t, pos));
        std::string rest = pos < t.size() ? t.substr(pos) : "";

        if (kw == "name") {
            out.name = trim(rest);
            out.params.model_alias = out.name;
        } else if (kw == "from") {
            out.params.model = trim(rest);
            saw_from = true;
        } else if (kw == "template") {
            out.template_text = consume_value(i, rest);
        } else if (kw == "system") {
            out.system_text = consume_value(i, rest);
        } else if (kw == "parameter") {
            auto parts = split_n(rest, 2);
            if (parts.size() < 2) {
                err = "PARAMETER needs `key value` at " + path + ":" + std::to_string(i + 1);
                return false;
            }
            if (!apply_param(out.params, parts[0], trim(parts[1]))) {
                err = "unknown PARAMETER key `" + parts[0] + "` at " + path + ":" + std::to_string(i + 1);
                return false;
            }
        } else if (kw == "tag") {
            std::string tag = trim(rest);
            if (!tag.empty()) out.tags.push_back(tag);
        } else if (kw == "speculative_model") {
            out.speculative.model = trim(rest);
        } else if (kw == "speculative_max") {
            try { out.speculative.max_draft = std::stoi(trim(rest)); }
            catch (...) { err = "SPECULATIVE_MAX expects int"; return false; }
        } else if (kw == "speculative_min") {
            try { out.speculative.min_draft = std::stoi(trim(rest)); }
            catch (...) { err = "SPECULATIVE_MIN expects int"; return false; }
        } else if (kw == "hardware") {
            auto parts = split_n(rest, 3);
            if (parts.size() < 3) {
                err = "HARDWARE needs `tier key value` at " + path + ":" + std::to_string(i + 1);
                return false;
            }
            int low, high;
            if (!parse_hw_tier_name(parts[0], low, high)) {
                err = "bad HARDWARE tier `" + parts[0] + "` at " + path + ":" + std::to_string(i + 1);
                return false;
            }
            // Find or create the tier
            hw_tier * tier = nullptr;
            for (auto & tt : out.hw_tiers) {
                if (tt.low_gb == low && tt.high_gb == high) { tier = &tt; break; }
            }
            if (!tier) {
                hw_tier nt;
                nt.low_gb = low;
                nt.high_gb = high;
                out.hw_tiers.push_back(nt);
                tier = &out.hw_tiers.back();
            }
            tier->params[parts[1]] = trim(parts[2]);
        } else {
            err = "unknown keyword `" + kw + "` at " + path + ":" + std::to_string(i + 1);
            return false;
        }
    }

    if (!saw_from || out.params.model.empty()) {
        err = "Modelfile is missing FROM <path>";
        return false;
    }
    return true;
}

void model_def_apply_hw(model_def & def, const hw_info & hw) {
    // Pick the single tier whose [low, high) covers hw.vram_gb. If multiple
    // match, prefer the one with the highest `low` (most specific).
    const hw_tier * best = nullptr;
    for (const auto & t : def.hw_tiers) {
        if (hw.vram_gb >= t.low_gb && hw.vram_gb < t.high_gb) {
            if (!best || t.low_gb > best->low_gb) best = &t;
        }
    }
    if (!best) return;
    for (const auto & kv : best->params) {
        apply_param(def.params, kv.first, kv.second);
    }
}

hw_info hw_detect() {
    hw_info hw;

    // GPU: use ggml's own enumeration when compiled with CUDA. The accessor
    // does a one-time probe with no cuBLAS init, so this is cheap to call
    // from the manager at startup. We surface the primary device (highest-
    // VRAM device wins, mirroring what tensor_split would default to).
#ifdef GGML_USE_CUBLAS
    const int n = ggml_cuda_get_device_count();
    int best = -1;
    size_t best_mem = 0;
    for (int i = 0; i < n; ++i) {
        const size_t m = ggml_cuda_get_device_memory(i);
        if (m > best_mem) { best_mem = m; best = i; }
    }
    if (best >= 0) {
        hw.vram_gb  = (int) (best_mem / (1024ULL * 1024ULL * 1024ULL));
        hw.gpu_name = ggml_cuda_get_device_name(best);
    }
#endif

    // RAM via sysconf
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        hw.ram_gb = (int) ((long long) pages * page_size / (1024LL * 1024LL * 1024LL));
    }

    return hw;
}

std::vector<std::string> scan_modelfiles(const std::string & dir) {
    std::vector<std::string> out;
    DIR * d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        if (!ends_with(name, ".modelfile")) continue;
        std::string full = dir;
        if (!full.empty() && full.back() != '/') full += '/';
        full += name;
        // skip non-regular files
        struct stat st;
        if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            out.push_back(full);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

static std::string json_escape(const std::string & s) {
    std::string r;
    r.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n"; break;
            case '\r': r += "\\r"; break;
            case '\t': r += "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    r += buf;
                } else {
                    r += c;
                }
        }
    }
    return r;
}

std::string model_def_to_json(const model_def & def) {
    std::ostringstream o;
    o << "{";
    o << "\"name\":\""  << json_escape(def.name) << "\",";
    o << "\"model\":\"" << json_escape(def.params.model) << "\",";
    o << "\"source\":\"" << json_escape(def.source_path) << "\",";
    o << "\"n_ctx\":"        << def.params.n_ctx        << ",";
    o << "\"n_gpu_layers\":" << def.params.n_gpu_layers << ",";
    o << "\"n_batch\":"      << def.params.n_batch      << ",";
    o << "\"temp\":"         << def.params.temp         << ",";
    o << "\"top_p\":"        << def.params.top_p        << ",";
    o << "\"tags\":[";
    for (size_t i = 0; i < def.tags.size(); ++i) {
        if (i) o << ",";
        o << "\"" << json_escape(def.tags[i]) << "\"";
    }
    o << "]";
    if (!def.speculative.model.empty()) {
        o << ",\"speculative\":{"
          << "\"model\":\"" << json_escape(def.speculative.model) << "\","
          << "\"max\":" << def.speculative.max_draft << ","
          << "\"min\":" << def.speculative.min_draft
          << "}";
    }
    o << "}";
    return o.str();
}
