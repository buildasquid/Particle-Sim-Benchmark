#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
const static Vector2d G(0.f, -10.f);   // external (gravitational) forces
const static float REST_DENS = 300.f;  // rest density
const static float GAS_CONST = 2000.f; // equation of state constant
const static float H = 16.f;           // kernel radius
const static float HSQ = H * H;        // radius^2 for optimization
const static float MASS = 2.5f;        // particle mass
const static float VISC = 200.f;       // viscosity constant
const static float DT = 0.0007f;       // integration timestep

// smoothing kernels
const static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
const static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
const static float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));

// simulation parameters
const static float EPS = H;            // boundary epsilon
const static float BOUND_DAMPING = -0.5f;

// particle structure
struct Particle {
    Particle(float _x, float _y) : x(_x, _y), v(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f) {}
    Vector2d x, v, f;
    float rho, p;
};

// simulation data
static vector<Particle> particles;

// interaction
const static int MAX_PARTICLES = 2500;
const static int DAM_PARTICLES = 500;
const static int BLOCK_PARTICLES = 250;

// rendering projection
const static int WINDOW_WIDTH = 800;
const static int WINDOW_HEIGHT = 600;
const static double VIEW_WIDTH = 1.5 * 800.f;
const static double VIEW_HEIGHT = 1.5 * 600.f;

// random number generator for Linux
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.f, 1.f);

void InitSPH() {
    cout << "Initializing dam break with " << DAM_PARTICLES << " particles" << endl;
    for (float y = EPS; y < VIEW_HEIGHT - EPS * 2.f; y += H) {
        for (float x = VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H) {
            if (particles.size() < DAM_PARTICLES) {
                float jitter = dist(gen); // replaces arc4random
                particles.push_back(Particle(x + jitter, y));
            } else {
                return;
            }
        }
    }
}

void Integrate() {
    for (auto &p : particles) {
        p.v += DT * p.f / p.rho;
        p.x += DT * p.v;

        // boundary
        if (p.x(0) - EPS < 0.f) { p.v(0) *= BOUND_DAMPING; p.x(0) = EPS; }
        if (p.x(0) + EPS > VIEW_WIDTH) { p.v(0) *= BOUND_DAMPING; p.x(0) = VIEW_WIDTH - EPS; }
        if (p.x(1) - EPS < 0.f) { p.v(1) *= BOUND_DAMPING; p.x(1) = EPS; }
        if (p.x(1) + EPS > VIEW_HEIGHT) { p.v(1) *= BOUND_DAMPING; p.x(1) = VIEW_HEIGHT - EPS; }
    }
}

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

void Update() {
    ComputeDensityPressure();
    ComputeForces();
    Integrate();
    glutPostRedisplay();
}

void InitGL() {
    glClearColor(0.9f, 0.9f, 0.9f, 1);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(H / 2.f);
    glMatrixMode(GL_PROJECTION);
}

void Render() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);
    glColor4f(0.2f, 0.6f, 1.f, 1);
    glBegin(GL_POINTS);
    for (auto &p : particles) glVertex2f(p.x(0), p.x(1));
    glEnd();
    glutSwapBuffers();
}

void Keyboard(unsigned char c, int x, int y) {
    switch (c) {
        case ' ':
            if (particles.size() >= MAX_PARTICLES) {
                cout << "Maximum number of particles reached" << endl;
            } else {
                unsigned int placed = 0;
                for (float y = VIEW_HEIGHT / 1.5f - VIEW_HEIGHT / 5.f; y < VIEW_HEIGHT / 1.5f + VIEW_HEIGHT / 5.f; y += H * 0.95f) {
                    for (float x = VIEW_WIDTH / 2.f - VIEW_HEIGHT / 5.f; x <= VIEW_WIDTH / 2.f + VIEW_HEIGHT / 5.f; x += H * 0.95f) {
                        if (placed++ < BLOCK_PARTICLES && particles.size() < MAX_PARTICLES) {
                            particles.push_back(Particle(x, y));
                        }
                    }
                }
            }
            break;
        case 'r':
        case 'R':
            particles.clear();
            InitSPH();
            break;
    }
}

int main(int argc, char **argv) {
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInit(&argc, argv);
    glutCreateWindow("Müller SPH");
    glutDisplayFunc(Render);
    glutIdleFunc(Update);
    glutKeyboardFunc(Keyboard);

    InitGL();
    InitSPH();

    glutMainLoop();
    return 0;
}
