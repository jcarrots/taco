#pragma once

#include <chrono>
#include <iomanip>
#include <ostream>
#include <string>
#include <utility>

namespace taco::profile {

class Session {
  public:
    explicit Session(bool enabled, std::ostream& os) : enabled_(enabled), os_(&os) {}

    bool enabled() const noexcept { return enabled_; }

    class Section {
      public:
        Section() = default;
        Section(Session* session, std::string name)
            : session_(session), name_(std::move(name)) {
            if (!session_ || !session_->enabled_) return;
            depth_ = session_->depth_++;
            start_ = clock::now();
            active_ = true;
        }

        Section(const Section&) = delete;
        Section& operator=(const Section&) = delete;

        Section(Section&& other) noexcept { *this = std::move(other); }
        Section& operator=(Section&& other) noexcept {
            if (this == &other) return *this;
            stop();
            session_ = other.session_;
            name_ = std::move(other.name_);
            start_ = other.start_;
            depth_ = other.depth_;
            active_ = other.active_;
            other.session_ = nullptr;
            other.active_ = false;
            return *this;
        }

        ~Section() { stop(); }

        void stop() noexcept {
            if (!active_ || !session_) return;
            const auto end = clock::now();
            const auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
            session_->depth_--;
            session_->print_line(depth_, name_, dur);
            active_ = false;
        }

      private:
        using clock = std::chrono::steady_clock;

        Session* session_{nullptr};
        std::string name_;
        clock::time_point start_{};
        int depth_{0};
        bool active_{false};
    };

    Section section(std::string name) { return Section(this, std::move(name)); }

  private:
    void print_line(int depth,
                    const std::string& name,
                    std::chrono::nanoseconds dur) const {
        if (!os_) return;
        const auto flags = os_->flags();
        const auto prec = os_->precision();
        const double ms = std::chrono::duration<double, std::milli>(dur).count();
        (*os_) << std::string(static_cast<std::size_t>(depth) * 2U, ' ')
               << name << ": " << std::fixed << std::setprecision(3) << ms << " ms\n";
        os_->flags(flags);
        os_->precision(prec);
    }

    bool enabled_{false};
    std::ostream* os_{nullptr};
    int depth_{0};
};

} // namespace taco::profile
