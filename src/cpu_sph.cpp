#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <filesystem>
namespace fs = std::filesystem;


using namespace std;
using namespace Eigen;


// Solver parameters
const static Vector2d G(0.f, -10.f);
const static float REST_DENS = 300.f;
const static float GAS_CONST = 2000.f;
const static float H = 16.f;
const static float HSQ = H * H;
const static float MASS = 2.5f;
const static float VISC = 200.f;
const static float DT = 0.0007f;
const static float EPS = H;
const static float BOUND_DAMPING = -0.5f;

// Smoothing kernels
const static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
const static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
const static float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));

// Particle structure
struct Particle {
    Particle(float _x, float _y) : x(_x, _y), v(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f) {}
    Vector2d x, v, f;
    float rho, p;
};

// Simulation data
static vector<Particle> particles;

// Particle counts
const static int MAX_PARTICLES = 2500;
const static int DAM_PARTICLES = 500;
const static int BLOCK_PARTICLES = 250;

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.f, 1.f);

// Initialize particles for dam break
void InitSPH() {
    particles.clear();
    cout << "Initializing dam break with " << DAM_PARTICLES << " particles" << endl;
    for (float y = EPS; y < 600.f - EPS * 2.f; y += H) {
        for (float x = 800.f / 4; x <= 800.f / 2; x += H) {
            if (particles.size() < DAM_PARTICLES) {
                float jitter = dist(gen);
                particles.push_back(Particle(x + jitter, y));
            } else return;
        }
    }
}

// Compute density and pressure
void ComputeDensityPressure() {
    for (auto &pi : particles) {
        pi.rho = 0.f;
        for (auto &pj : particles) {
            Vector2d rij = pj.x - pi.x;
            float r2 = rij.squaredNorm();
            if (r2 < HSQ) {
                pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
            }
        }
        pi.p = GAS_CONST * (pi.rho - REST_DENS);
    }
}

// Compute forces
void ComputeForces() {
    for (auto &pi : particles) {
        Vector2d fpress(0.f, 0.f);
        Vector2d fvisc(0.f, 0.f);
        for (auto &pj : particles) {
            if (&pi == &pj) continue;
            Vector2d rij = pj.x - pi.x;
            float r = rij.norm();
            if (r < H) {
                fpress += -rij.normalized() * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
                fvisc += VISC * MASS * (pj.v - pi.v) / pj.rho * VISC_LAP * (H - r);
            }
        }
        Vector2d fgrav = G * MASS / pi.rho;
        pi.f = fpress + fvisc + fgrav;
    }
}

// Integrate positions and velocities
void Integrate() {
    for (auto &p : particles) {
        p.v += DT * p.f / p.rho;
        p.x += DT * p.v;

        // boundary conditions
        if (p.x(0) - EPS < 0.f) { p.v(0) *= BOUND_DAMPING; p.x(0) = EPS; }
        if (p.x(0) + EPS > 800.f) { p.v(0) *= BOUND_DAMPING; p.x(0) = 800.f - EPS; }
        if (p.x(1) - EPS < 0.f) { p.v(1) *= BOUND_DAMPING; p.x(1) = EPS; }
        if (p.x(1) + EPS > 600.f) { p.v(1) *= BOUND_DAMPING; p.x(1) = 600.f - EPS; }
    }
}

// Single simulation step
void Step() {
    ComputeDensityPressure();
    ComputeForces();
    Integrate();
}


// Save CSV snapshot
void SaveParticlesCSV(int step) {
    fs::create_directories("/home/daria/Thesis/Particle-Sim-Benchmark/data");
    std::ofstream file("/home/daria/Thesis/Particle-Sim-Benchmark/data/particles_step_" + std::to_string(step) + ".csv");

    file << "x,y,vx,vy,rho,p\n";
    for (auto &p : particles) {
        file << fixed << setprecision(5)
             << p.x(0) << "," << p.x(1) << ","
             << p.v(0) << "," << p.v(1) << ","
             << p.rho << "," << p.p << "\n";
    }
}

// Main benchmark
int main() {
    InitSPH();

    const int NUM_STEPS = 10000;
    const int SAVE_EVERY = 1000;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_STEPS; ++i) {
        Step();
        if (i % SAVE_EVERY == 0) {
            SaveParticlesCSV(i);
            cout << "Saved step " << i << " to CSV." << endl;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();
    cout << "Total simulation time: " << elapsed << " seconds." << endl;
    cout << "Average step time: " << (elapsed / NUM_STEPS) << " seconds." << endl;
}