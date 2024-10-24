#include <iostream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <cmath>
#include <random>
#include <chrono>
#include <utils.h>
#include <lerw_2d.cpp>
#include <lerw_3d.cpp>

using namespace std;

int main() {
    // Parameters
    vector<double> R_values = {10, 20, 40, 80, 160};
    int num_trials = 1000;

    // Stores for lengths
    vector<double> avg_lengths_2D(R_values.size(), 0.0);
    vector<double> avg_lengths_3D(R_values.size(), 0.0);

    // Simulation for 2D
    cout << "Simulating LERW in 2D..." << endl;
    for (size_t i = 0; i < R_values.size(); ++i) {
        double total_length = 0.0;
        double R = R_values[i];
        for (int j = 0; j < num_trials; ++j) {
            vector<Vec2D> path = simulate_LERW_2D(R);
            total_length += path.size();
        }
        avg_lengths_2D[i] = total_length / num_trials;
        cout << "R = " << R << ", Avg Length = " << avg_lengths_2D[i] << endl;
    }

    // Simulation for 3D
    cout << "\nSimulating LERW in 3D..." << endl;
    for (size_t i = 0; i < R_values.size(); ++i) {
        double total_length = 0.0;
        double R = R_values[i];
        for (int j = 0; j < num_trials; ++j) {
            vector<Vec3D> path = simulate_LERW_3D(R);
            total_length += path.size();
        }
        avg_lengths_3D[i] = total_length / num_trials;
        cout << "R = " << R << ", Avg Length = " << avg_lengths_3D[i] << endl;
    }

    // Estimating the exponent D
    cout << "\nEstimating the exponent D..." << endl;
    double sum_logR = 0.0, sum_logL_2D = 0.0, sum_logL_3D = 0.0;
    double sum_logR2 = 0.0, sum_logR_logL_2D = 0.0, sum_logR_logL_3D = 0.0;
    int n = R_values.size();

    for (int i = 0; i < n; ++i) {
        double logR = log(R_values[i]);
        double logL_2D = log(avg_lengths_2D[i]);
        double logL_3D = log(avg_lengths_3D[i]);

        sum_logR += logR;
        sum_logL_2D += logL_2D;
        sum_logL_3D += logL_3D;
        sum_logR2 += logR * logR;
        sum_logR_logL_2D += logR * logL_2D;
        sum_logR_logL_3D += logR * logL_3D;
    }

    double D_2D = (n * sum_logR_logL_2D - sum_logR * sum_logL_2D) / (n * sum_logR2 - sum_logR * sum_logR);
    double D_3D = (n * sum_logR_logL_3D - sum_logR * sum_logL_3D) / (n * sum_logR2 - sum_logR * sum_logR);

    cout << "Estimated exponent D in 2D: " << D_2D << endl;
    cout << "Estimated exponent D in 3D: " << D_3D << endl;

    return 0;
}
