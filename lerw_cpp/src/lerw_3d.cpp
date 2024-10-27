// lerw_3d.cpp
#include "lerw_3d.h"
#include <unordered_map>
#include <cmath>
#include <random>

std::vector<Vec3D> simulate_LERW_3D(double R_max, std::mt19937& rng) {
    std::unordered_map<Vec3D, int> visited;
    std::vector<Vec3D> path;
    Vec3D current = {0, 0, 0};
    path.push_back(current);
    visited[current] = 0;

    // Create a distribution using the passed RNG
    std::uniform_int_distribution<int> direction_dist(0, 5);

    while (true) {
        // Choose a random direction using the thread's RNG
        int dir = direction_dist(rng);
        Vec3D next = current;

        switch (dir) {
            case 0: next.x += 1; break; // Right
            case 1: next.x -= 1; break; // Left
            case 2: next.y += 1; break; // Up
            case 3: next.y -= 1; break; // Down
            case 4: next.z += 1; break; // Forward
            case 5: next.z -= 1; break; // Backward
        }

        // Check if next position is already visited
        if (visited.find(next) != visited.end()) {
            // Loop detected; erase the loop
            int loop_start = visited[next];
            for (int i = path.size() - 1; i >= loop_start; --i) {
                visited.erase(path[i]);
                path.pop_back();
            }
            current = next;
            path.push_back(current);
            visited[current] = path.size() - 1;
        } else {
            // No loop; proceed normally
            current = next;
            path.push_back(current);
            visited[current] = path.size() - 1;
        }

        // Check if the Euclidean distance exceeds R_max
        double distance = sqrt(current.x * current.x + current.y * current.y + current.z * current.z);
        if (distance >= R_max) {
            break;
        }
    }

    return path;
}