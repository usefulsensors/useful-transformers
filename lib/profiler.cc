#include "profiler.h"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

Profiler::Profiler(bool enable)
    : _enabled(enable),
      _running(false),
      _run_count(0),
      _total_duration(std::chrono::microseconds::zero()) {}

Profiler::~Profiler() {
  if (!_enabled) {
    return;
  }
  std::vector<std::pair<std::string, std::chrono::microseconds>> sorted;
  for (auto& x : _profiles) {
    sorted.push_back(x);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](std::pair<std::string, std::chrono::microseconds> a,
               std::pair<std::string, std::chrono::microseconds> b) -> bool {
              return a.second.count() > b.second.count();
            });

  std::cout << std::left;
  std::cout << std::setw(32) << "Name";
  std::cout << std::setw(32) << "Duration (ms)";
  std::cout << "Overall (%)\n";
  for (auto& x : sorted) {
    auto tag_duration = x.second.count();
    auto total_duration = _total_duration.count();
    float percentage =
        static_cast<float>((1e4 * tag_duration) / total_duration) / 100.0;
    float duration_ms = static_cast<float>(tag_duration / _run_count) / 1000.0;
    std::cout << std::left << std::fixed << std::setprecision(3);
    std::cout << std::setw(32) << x.first;
    std::cout << std::setw(32) << duration_ms;
    std::cout << std::setw(32) << percentage;
    std::cout << "\n";
  }
}

void Profiler::start() {
  if (!_enabled) {
    return;
  }
  _last = std::chrono::system_clock::now();
  _running = true;
  _run_count++;
}

void Profiler::pause() {
  _running = false;
  if (!_enabled) {
    return;
  }
}

void Profiler::measure(std::string tag) {
  if (!_enabled) {
    return;
  }
  assert(_running);
  auto last_saved = _last;
  _last = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(_last - last_saved);
  _profiles[tag] += duration;
  _total_duration += duration;
}
