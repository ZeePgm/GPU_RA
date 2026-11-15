#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace hbflab {
enum class ReadonlyStrategy { PtxScalar, CudaWarp, CudaScalar };

// Host-side policy and accounting shared by both readonly launch adapters.
// Borrowed byte storage and the timing service retain their existing owners.
class ReadonlyAccess {
public:
  struct Entry {
    std::string name;
    std::size_t bytes;
    ReadonlyStrategy strategy;
    const void *address = nullptr;
    std::uint64_t launches = 0, requests = 0;
    unsigned expected = 0, handled = 0;
  };
  void add(std::string name, std::size_t bytes, ReadonlyStrategy strategy) {
    current();
    if (started_ || name.empty() || !bytes ||
        (strategy != ReadonlyStrategy::PtxScalar && strategy != ReadonlyStrategy::CudaWarp &&
         strategy != ReadonlyStrategy::CudaScalar))
      throw std::invalid_argument("readonly registration invalid");
    for (const auto &e : entries_)
      if (e.name == name)
        throw std::invalid_argument("duplicate readonly name");
    entries_.push_back({std::move(name), bytes, strategy});
  }
  void bind(std::size_t index, const void *address) {
    current();
    auto &e = entries_.at(index);
    const auto p = reinterpret_cast<std::uintptr_t>(address);
    if (started_ || e.address || !p ||
        e.bytes > std::numeric_limits<std::uintptr_t>::max() - p)
      throw std::invalid_argument("readonly binding invalid");
    for (const auto &other : entries_)
      if (other.address) {
        const auto b = reinterpret_cast<std::uintptr_t>(other.address);
        if (p >= b ? p - b < other.bytes : b - p < e.bytes)
          throw std::invalid_argument("readonly ranges overlap");
      }
    e.address = address;
  }
  void start() {
    current();
    if (started_ || entries_.empty())
      throw std::logic_error("readonly start invalid");
    for (const auto &e : entries_)
      if (!e.address)
        throw std::logic_error("readonly unbound range");
    started_ = true;
  }
  int find(const char *name, const void *address, std::size_t bytes) const {
    current();
    for (std::size_t i = 0; i < entries_.size(); ++i)
      if (name && entries_[i].name == name)
        return static_cast<int>(i);
    const auto p = reinterpret_cast<std::uintptr_t>(address);
    if (p && bytes)
      for (std::size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].address) {
          const auto b = reinterpret_cast<std::uintptr_t>(entries_[i].address);
          if (p >= b ? p - b < entries_[i].bytes : b - p < bytes)
            return static_cast<int>(i);
        }
    return -1;
  }
  void begin_step(int step) {
    active();
    if (in_step_ || step != step_ + 1)
      throw std::logic_error("readonly step sequence invalid");
    step_ = step;
    in_step_ = true;
    for (auto &e : entries_)
      e.expected = e.handled = 0;
  }
  void expect(std::size_t index) {
    step();
    ++entries_.at(index).expected;
  }
  void handled(std::size_t index) {
    step();
    ++entries_.at(index).handled;
  }
  template <class Launch>
  auto dispatch(std::size_t index, ReadonlyStrategy strategy,
                const void *address, std::size_t bytes, std::uint64_t requests,
                Launch launch) -> decltype(launch()) {
    step();
    auto &e = entries_.at(index);
    const auto p = reinterpret_cast<std::uintptr_t>(address),
               b = reinterpret_cast<std::uintptr_t>(e.address);
    if (e.strategy != strategy || p < b || !bytes || bytes > e.bytes ||
        p - b > e.bytes - bytes)
      throw std::invalid_argument("readonly dispatch role/range invalid");
    try {
      auto result = launch();
      ++e.launches;
      e.requests += requests;
      return result;
    } catch (...) {
      failed_ = true;
      throw;
    }
  }
  void end_step(bool require_coverage) {
    step();
    for (const auto &e : entries_)
      if (require_coverage && (!e.expected || e.expected != e.handled))
        throw std::logic_error("readonly selected coverage incomplete");
    in_step_ = false;
  }
  void finish() {
    active();
    if (in_step_)
      throw std::logic_error("readonly unfinished step");
    closed_ = true;
  }
  void abort() noexcept { failed_ = true; }
  const std::vector<Entry> &entries() const { return entries_; }
  std::vector<unsigned> expected() const {
    std::vector<unsigned> v;
    for (const auto &e : entries_)
      v.push_back(e.expected);
    return v;
  }
  std::vector<unsigned> handled() const {
    std::vector<unsigned> v;
    for (const auto &e : entries_)
      v.push_back(e.handled);
    return v;
  }

private:
  void current() const {
    if (owner_ != std::this_thread::get_id() || closed_ || failed_)
      throw std::logic_error("readonly owner/lifecycle invalid");
  }
  void active() const {
    current();
    if (!started_)
      throw std::logic_error("readonly not started");
  }
  void step() const {
    active();
    if (!in_step_)
      throw std::logic_error("readonly no active step");
  }
  std::thread::id owner_ = std::this_thread::get_id();
  std::vector<Entry> entries_;
  bool started_ = false, in_step_ = false, closed_ = false, failed_ = false;
  int step_ = -1;
};
} // namespace hbflab
