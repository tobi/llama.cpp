// llama.cpp model orchestrator.
//
// Watches a directory of Modelfiles, picks the right one to load based on
// detected hardware and a list of autoload tags, talks to a running
// `./server` over HTTP (POST /load, /unload, GET /slots), and unloads the
// model when it has been idle for too long.
//
// Manager config is itself a small line-based file (`manager.conf`):
//
//   server_url            http://127.0.0.1:8080
//   models_root           ~/.config/llama/models
//   idle_timeout_seconds  120
//   poll_interval_seconds 5
//   autoload              true
//   autoload_tag          default
//   autoload_tag          coding         # repeat for multiple
//
// All keys are optional; defaults match the spec.

#define CPPHTTPLIB_THREAD_POOL_COUNT 1
#include "httplib.h"
#include "json.hpp"
#include "model-config.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <pwd.h>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

struct manager_cfg {
    std::string server_url     = "http://127.0.0.1:8080";
    std::string models_root    = "~/.config/llama/models";
    int  idle_timeout_seconds  = 120;
    int  poll_interval_seconds = 5;
    bool autoload              = true;
    std::vector<std::string> autoload_tags;  // empty = any
};

static std::string expand_home(const std::string & path) {
    if (path.empty() || path[0] != '~') return path;
    const char * home = getenv("HOME");
    if (!home) {
        struct passwd * pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (!home) return path;
    return std::string(home) + path.substr(1);
}

static bool load_manager_cfg(const std::string & path, manager_cfg & cfg) {
    std::ifstream f(path);
    if (!f) return false;
    std::string raw;
    while (std::getline(f, raw)) {
        std::string line = trim(raw);
        if (line.empty() || line[0] == '#') continue;
        // strip trailing inline comment
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = trim(line.substr(0, hash));
        size_t sp = line.find_first_of(" \t");
        if (sp == std::string::npos) continue;
        std::string key = line.substr(0, sp);
        std::string val = trim(line.substr(sp + 1));

        if      (key == "server_url")            cfg.server_url = val;
        else if (key == "models_root")           cfg.models_root = val;
        else if (key == "idle_timeout_seconds")  cfg.idle_timeout_seconds = std::stoi(val);
        else if (key == "poll_interval_seconds") cfg.poll_interval_seconds = std::stoi(val);
        else if (key == "autoload") {
            std::string v = val;
            std::transform(v.begin(), v.end(), v.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            cfg.autoload = (v == "true" || v == "1" || v == "yes" || v == "on");
        }
        else if (key == "autoload_tag")          cfg.autoload_tags.push_back(val);
        else {
            fprintf(stderr, "manager: unknown key `%s` in %s\n", key.c_str(), path.c_str());
        }
    }
    return true;
}

// Split a URL like http://host:port into (host, port) for httplib::Client.
static bool parse_url(const std::string & url, std::string & host, int & port, bool & https) {
    https = false;
    std::string s = url;
    if (s.rfind("http://", 0) == 0)       { s.erase(0, 7); https = false; }
    else if (s.rfind("https://", 0) == 0) { s.erase(0, 8); https = true;  }
    size_t slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    size_t colon = s.find(':');
    if (colon != std::string::npos) {
        host = s.substr(0, colon);
        try { port = std::stoi(s.substr(colon + 1)); }
        catch (...) { return false; }
    } else {
        host = s;
        port = https ? 443 : 80;
    }
    return !host.empty();
}

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop.store(true); }

// True if `def` carries any of the requested tags (or if the request list is empty).
static bool matches_autoload(const model_def & def, const std::vector<std::string> & want) {
    if (want.empty()) return true;
    for (const auto & w : want) {
        for (const auto & t : def.tags) {
            if (t == w) return true;
        }
    }
    return false;
}

// Pick the best Modelfile to autoload from a list of paths. "Best" =
// (a) matches an autoload tag, (b) fits in current VRAM (model size unknown
// here, so we proxy with n_gpu_layers > 0 only when we have VRAM), (c) the
// alphabetically-first qualifying path wins for determinism. Returns the
// loaded def by reference; result is the chosen path or "" if none qualify.
static std::string pick_autoload(const std::vector<std::string> & paths,
                                 const std::vector<std::string> & want_tags,
                                 const hw_info & hw,
                                 model_def & chosen) {
    for (const auto & p : paths) {
        model_def def;
        gpt_params defaults;
        std::string err;
        if (!model_def_load(p, defaults, def, err)) {
            fprintf(stderr, "manager: skip %s (%s)\n", p.c_str(), err.c_str());
            continue;
        }
        model_def_apply_hw(def, hw);
        if (!matches_autoload(def, want_tags)) continue;
        if (def.params.n_gpu_layers > 0 && hw.vram_gb == 0) {
            // Wants GPU but we have none — skip.
            continue;
        }
        chosen = def;
        return p;
    }
    return "";
}

// Log a single line of NDJSON. `detail` is treated as already-formed JSON;
// use `log_event_s` if you have an arbitrary string that needs escaping.
static void log_event(const std::string & msg, const std::string & detail_json = "") {
    const int64_t t = (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (detail_json.empty())
        fprintf(stdout, "{\"ts\":%lld,\"msg\":\"%s\"}\n", (long long) t, msg.c_str());
    else
        fprintf(stdout, "{\"ts\":%lld,\"msg\":\"%s\",\"detail\":%s}\n",
                (long long) t, msg.c_str(), detail_json.c_str());
    fflush(stdout);
}

static void log_event_s(const std::string & msg, const std::string & detail_str) {
    log_event(msg, json(detail_str).dump());
}

int main(int argc, char ** argv) {
    manager_cfg cfg;
    std::string cfg_path = argc > 1 ? argv[1] : expand_home("~/.config/llama/manager.conf");

    if (!load_manager_cfg(cfg_path, cfg)) {
        log_event_s("config-not-found-using-defaults", cfg_path);
    } else {
        log_event_s("config-loaded", cfg_path);
    }
    cfg.models_root = expand_home(cfg.models_root);

    // Parse server URL once.
    std::string host;
    int  port = 0;
    bool https = false;
    if (!parse_url(cfg.server_url, host, port, https)) {
        fprintf(stderr, "manager: bad server_url `%s`\n", cfg.server_url.c_str());
        return 1;
    }
    if (https) {
        fprintf(stderr, "manager: https is not supported in this build\n");
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    hw_info hw = hw_detect();
    {
        std::ostringstream o;
        o << "{\"vram_gb\":" << hw.vram_gb
          << ",\"ram_gb\":" << hw.ram_gb
          << ",\"gpu\":\"" << hw.gpu_name << "\"}";
        log_event("hardware-detected", o.str());
    }

    httplib::Client cli(host, port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(5, 0);

    std::string loaded_alias;        // alias the server reports as loaded, or ""
    std::string loaded_source;       // source Modelfile path for the loaded model

    // Re-scan and reconcile on every tick.
    while (!g_stop.load()) {
        // 1) Ask the server what it has loaded right now.
        bool server_up = false;
        bool srv_loaded = false;
        int64_t srv_idle_ms = 0;
        std::string srv_alias, srv_source;
        {
            auto r = cli.Get("/slots");
            if (r) {
                server_up = true;
                try {
                    auto j = json::parse(r->body);
                    srv_loaded  = j.value("loaded", false);
                    srv_alias   = j.value("alias", "");
                    srv_source  = j.value("source", "");
                    srv_idle_ms = j.value("idle_ms", (int64_t) 0);
                } catch (const std::exception & e) {
                    log_event_s("slots-parse-error", e.what());
                }
            } else {
                log_event_s("server-down", "err=" + std::to_string((int) r.error()));
            }
        }

        if (!server_up) {
            std::this_thread::sleep_for(std::chrono::seconds(cfg.poll_interval_seconds));
            continue;
        }

        loaded_alias  = srv_loaded ? srv_alias  : "";
        loaded_source = srv_loaded ? srv_source : "";

        // 2) Idle unload
        if (srv_loaded && cfg.idle_timeout_seconds > 0 &&
            srv_idle_ms / 1000 >= (int64_t) cfg.idle_timeout_seconds) {
            log_event_s("idle-unload", loaded_alias);
            cli.Post("/unload", "{}", "application/json");
            loaded_alias.clear();
            loaded_source.clear();
            srv_loaded = false;
        }

        // 3) Autoload if nothing is loaded and we're configured to.
        if (!srv_loaded && cfg.autoload) {
            auto paths = scan_modelfiles(cfg.models_root);
            if (paths.empty()) {
                log_event_s("no-modelfiles", cfg.models_root);
            } else {
                model_def chosen;
                std::string p = pick_autoload(paths, cfg.autoload_tags, hw, chosen);
                if (!p.empty()) {
                    json req;
                    req["model_config"] = p;
                    auto r = cli.Post("/load", req.dump(), "application/json");
                    if (r && r->status == 200) {
                        log_event_s("autoload-ok", chosen.name);
                        loaded_alias  = chosen.name;
                        loaded_source = p;
                    } else {
                        std::string err = r ? r->body : std::string("(no response)");
                        log_event_s("autoload-failed", err);
                    }
                } else {
                    log_event("no-autoload-match");
                }
            }
        }

        // 4) Sleep until next tick, but wake early on signal.
        for (int i = 0; i < cfg.poll_interval_seconds * 10 && !g_stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    log_event("shutdown");
    return 0;
}
