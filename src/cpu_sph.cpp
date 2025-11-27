#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <sys/resource.h>
#include <unistd.h>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <fstream>


namespace fs = std::filesystem;
using namespace std;
using namespace Eigen;

// ---------------------- Solver parameters ----------------------
const static Vector2d G(0.f, -10.f);
const static float REST_DENS = 300.f;
const static float GAS_CONST = 2000.f;
const static float BOUND_DAMPING = -0.5f;

// ---------------------- Particle ----------------------
struct Particle {
    Particle(float _x, float _y) : x(_x, _y), v(0.f,0.f), f(0.f,0.f), rho(0), p(0.f) {}
    Vector2d x, v, f;
    float rho, p;
};

static vector<Particle> particles;

// Random generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.f, 1.f);

// ---------------------- Utility functions ----------------------
size_t GetCurrentRSS() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return r_usage.ru_maxrss * 1024L; // ru_maxrss in KB
}

double GetCPUPercent(clock_t cpu_time, double wall_time, int num_cores) {
    double cpu_sec = cpu_time / (double)CLOCKS_PER_SEC;
    return (cpu_sec / wall_time) * 100.0 / num_cores;
}

// ---------------------- SPH Initialization ----------------------
void InitSPH(int NUM_PARTICLES, float H, float EPS) {
    particles.clear();
    for (float y = EPS; y < 600.f - EPS * 2.f; y += H) {
        for (float x = 800.f/4; x <= 800.f/2; x += H) {
            if (particles.size() < NUM_PARTICLES) {
                float jitter = dist(gen);
                particles.push_back(Particle(x + jitter, y));
            } else return;
        }
    }
}

// ---------------------- SPH Simulation ----------------------
void ComputeDensityPressure(float MASS, float H, float HSQ, float POLY6, float REST_DENS, float GAS_CONST) {
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

void ComputeForces(float MASS, float H, float SPIKY_GRAD, float VISC_LAP, float VISC) {
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

void Integrate(float DT, float H, float EPS) {
    for (auto &p : particles) {
        p.v += DT * p.f / p.rho;
        p.x += DT * p.v;

        if (p.x(0) - EPS < 0.f) { p.v(0) *= BOUND_DAMPING; p.x(0) = EPS; }
        if (p.x(0) + EPS > 800.f) { p.v(0) *= BOUND_DAMPING; p.x(0) = 800.f - EPS; }
        if (p.x(1) - EPS < 0.f) { p.v(1) *= BOUND_DAMPING; p.x(1) = EPS; }
        if (p.x(1) + EPS > 600.f) { p.v(1) *= BOUND_DAMPING; p.x(1) = 600.f - EPS; }
    }
}

void Step(float MASS, float H, float HSQ, float POLY6, float SPIKY_GRAD, float VISC_LAP, float VISC, float DT, float REST_DENS, float GAS_CONST) {
    ComputeDensityPressure(MASS, H, HSQ, POLY6, REST_DENS, GAS_CONST);
    ComputeForces(MASS, H, SPIKY_GRAD, VISC_LAP, VISC);
    Integrate(DT, H, H);
}


int LoadResumeID() {
    ifstream file("resume_id.txt");
    int id = 0;
    if (file.is_open()) file >> id;
    return id;
}

void SaveResumeID(int id) {
    ofstream file("resume_id.txt", ios::trunc);
    file << id;
}


// ---------------------- CSV Output ----------------------
void SaveAllStepsCSV(const string &filename, int step, double step_time, size_t curr_mem, size_t peak_mem, int num_threads, int num_cores) {
    // Open file in append mode for each call
    ofstream file;
    if (step == 0) { // only write header at the first step
        fs::create_directories("/home/daria/Thesis/Particle-Sim-Benchmark/data");
        file.open(filename, ios::out);
        file << "step,particle_id,x,y,vx,vy,rho,p,step_time,mem_bytes,peak_mem_bytes,mem_per_particle,cpu_percent,num_threads,num_cores,thread_efficiency\n";
    } else {
        file.open(filename, ios::app);
    }

    double mem_per_particle = curr_mem / (double)particles.size();
    clock_t cpu_time = clock();
    double cpu_percent = GetCPUPercent(cpu_time, step_time, num_cores); // percent relative to CPU cores
    double thread_efficiency = cpu_percent / 100.0 * num_threads; // efficiency per thread

    for (size_t i = 0; i < particles.size(); ++i) {
        auto &p = particles[i];
        file << step << "," << i << "," << fixed << setprecision(5)
             << p.x(0) << "," << p.x(1) << "," << p.v(0) << "," << p.v(1) << "," 
             << p.rho << "," << p.p << "," << step_time << "," << curr_mem << "," << peak_mem << ","
             << mem_per_particle << "," << cpu_percent << "," << num_threads << "," << num_cores << "," << thread_efficiency << "\n";
    }
    file.close();
}


// ---------------------- Parameter Sweep ----------------------
void RunParameterSweepArrays(const vector<int> &particle_counts,
                             const vector<int> &num_steps_list,
                             const vector<float> &H_values,
                             const vector<float> &DT_values,
                             const vector<float> &VISC_values,
                             const vector<int> &thread_counts,
                             const vector<int> &cpu_cores_list,
                             float MASS,
                             float GAS_CONST) 
{
    int resume_from = LoadResumeID();
    int test_id = 0;

    for (auto NUM_PARTICLES : particle_counts) {
        for (auto NUM_STEPS : num_steps_list) {
            for (auto H_local : H_values) {
                float HSQ_local = H_local * H_local;
                float POLY6_local = 4.f / (M_PI * pow(H_local, 8.f));
                float SPIKY_GRAD_local = -10.f / (M_PI * pow(H_local, 5.f));
                float VISC_LAP_local = 40.f / (M_PI * pow(H_local, 5.f));

                for (auto DT_local : DT_values) {
                    for (auto VISC_local : VISC_values) {
                        for (auto NUM_THREADS : thread_counts) {
                            for (auto NUM_CORES : cpu_cores_list) {

                                // --------------------------
                                // SKIP UNTIL RESUME POINT
                                // --------------------------
                                if (test_id < resume_from) {
                                    cout << "Skipping test ID " << test_id << endl;
                                    test_id++;
                                    continue;
                                }

                                cout << "Running TEST ID " << test_id << " with "
                                     << NUM_PARTICLES << " particles, "
                                     << NUM_STEPS << " steps, H=" << H_local
                                     << ", DT=" << DT_local
                                     << ", VISC=" << VISC_local
                                     << ", threads=" << NUM_THREADS
                                     << ", CPU cores=" << NUM_CORES << endl;

                                InitSPH(NUM_PARTICLES, H_local, H_local);

                                string filename =
                                    "/home/daria/Thesis/Particle-Sim-Benchmark/data/particles_np" 
                                    + to_string(NUM_PARTICLES) + "_ns" + to_string(NUM_STEPS)
                                    + "_H" + to_string(H_local) + "_DT" + to_string(DT_local)
                                    + "_VISC" + to_string(VISC_local) + "_t" + to_string(NUM_THREADS)
                                    + "_cores" + to_string(NUM_CORES) + ".csv";

                                size_t peak_mem = 0;
                                auto sim_start = chrono::high_resolution_clock::now();

                                for (int step_idx = 0; step_idx < NUM_STEPS; ++step_idx) {
                                    auto step_start = chrono::high_resolution_clock::now();
                                    Step(MASS, H_local, HSQ_local, POLY6_local, SPIKY_GRAD_local,
                                         VISC_LAP_local, VISC_local, DT_local, REST_DENS, GAS_CONST);
                                    auto step_end = chrono::high_resolution_clock::now();
                                    double step_time = chrono::duration<double>(step_end - step_start).count();

                                    size_t curr_mem = GetCurrentRSS();
                                    if (curr_mem > peak_mem) peak_mem = curr_mem;

                                    SaveAllStepsCSV(filename, step_idx, step_time, curr_mem,
                                                    peak_mem, NUM_THREADS, NUM_CORES);
                                }

                                auto sim_end = chrono::high_resolution_clock::now();
                                double total_time = chrono::duration<double>(sim_end - sim_start).count();
                                cout << "Finished simulation in " << total_time << " s\n\n";

                                // --------------------------
                                // SAVE RESUME POINT
                                // --------------------------
                                SaveResumeID(test_id + 1);

                                test_id++;
                            }
                        }
                    }
                }
            }
        }
    }
}


// ---------------------- Main ----------------------
int main() {
    vector<int> PARTICLE_COUNTS = {100, 1000, 5000};               // small, medium, large
    vector<int> NUM_STEPS = {500, 2500, 10000};                    // enough steps to measure performance trends
    vector<float> H_VALUES = {8.0f, 16.0f};                 // small, medium, large smoothing lengths
    vector<float> DT_VALUES = {0.0005f, 0.0010f};                  // stable and slightly aggressive time steps
    vector<float> VISC_VALUES = {50.f, 500.f};             // low, medium, high viscosity
    vector<int> THREAD_COUNTS = {1, 2, 4};                         // representative threading
    vector<int> CPU_CORES = {1, 4, 8};                             // typical CPU core setups for testing
    float MASS = 2.5f;
    float GAS_CONST = 2000.f;                                       // total 486 simulations


    RunParameterSweepArrays(PARTICLE_COUNTS, NUM_STEPS, H_VALUES, DT_VALUES, VISC_VALUES, THREAD_COUNTS, CPU_CORES, MASS, GAS_CONST);
    return 0;
}
