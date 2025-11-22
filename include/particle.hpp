#pragma once
#include <array>
#include <vector>
using Vector2d = std::array<double, 2>;    
struct Particle {
    Vector2d x, v, f;
    float rho, p;
};


