#ifndef _LIB_PROFILER_H_
#define _LIB_PROFILER_H_

#include <chrono>
#include <string>
#include <unordered_map>

struct Profiler {
  bool _enabled;
  bool _running;
  int _run_count;
  std::chrono::microseconds _total_duration;
  std::chrono::time_point<std::chrono::system_clock> _last;
  std::unordered_map<std::string, std::chrono::microseconds> _profiles;

  Profiler(bool enable);
  ~Profiler();
  void start();
  void pause();
  void measure(std::string tag);
};

#endif  // _LIB_PROFILER_H_
