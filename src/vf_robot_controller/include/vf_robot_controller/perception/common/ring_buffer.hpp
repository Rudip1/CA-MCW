// Generic ring buffer for temporal state.
// Used by costmap-history, critic-cost-history, etc. Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__COMMON__RING_BUFFER_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__COMMON__RING_BUFFER_HPP_

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace vf_robot_controller::perception {

// Fixed-capacity ring buffer with newest-last semantics.
//   push(x)        — append, evicting oldest if full
//   size()         — current count, ≤ capacity
//   latest()       — most-recently pushed (size()-1)
//   oldest()       — earliest still retained (index 0)
//   at(i)          — i==0 is oldest, i==size()-1 is latest
template<typename T>
class RingBuffer {
public:
  explicit RingBuffer(size_t capacity)
  : capacity_(capacity == 0 ? 1 : capacity), data_(capacity_) {}

  void push(T value) {
    data_[head_] = std::move(value);
    head_ = (head_ + 1) % capacity_;
    if (size_ < capacity_) ++size_;
  }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }
  bool full() const { return size_ == capacity_; }
  bool empty() const { return size_ == 0; }

  // Index 0 = oldest, size()-1 = latest. Bounds-checked.
  const T & at(size_t i) const {
    if (i >= size_) throw std::out_of_range("RingBuffer::at");
    const size_t base = (head_ + capacity_ - size_) % capacity_;
    return data_[(base + i) % capacity_];
  }
  T & at(size_t i) {
    return const_cast<T &>(static_cast<const RingBuffer *>(this)->at(i));
  }

  const T & latest() const { return at(size_ - 1); }
  const T & oldest() const { return at(0); }

  void clear() {
    size_ = 0;
    head_ = 0;
  }

private:
  size_t capacity_;
  std::vector<T> data_;
  size_t head_{0};   // next write index
  size_t size_{0};
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__COMMON__RING_BUFFER_HPP_
