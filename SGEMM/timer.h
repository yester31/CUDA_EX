#include <chrono>
#include <string>
#include <iostream>

class Timer
{
public:
    Timer(const std::string &name = "Timer") : name_(name)
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        // Stop();
    }

    void Stop()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << ": " << duration.count() / 1000.0f << " ms" << std::endl;
    }

    float ElapsedMilliseconds() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0f;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};