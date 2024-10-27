// lerw_2d.cpp
#include "lerw_2d.h"
#include <unordered_map>
#include <cmath>
#include <random>

std::vector<Vec2D> simulate_LERW_2D(double R_max, std::mt19937& rng) {
    std::unordered_map<Vec2D, int> visited;
    std::vector<Vec2D> path;
    Vec2D current = {0, 0};
    path.push_back(current);
    visited[current] = 0;

    // Create a distribution using the passed RNG
    std::uniform_int_distribution<int> direction_dist(0, 3);

    while (true) {
        // Choose a random direction using the thread's RNG
        int dir = direction_dist(rng);
        Vec2D next = current;

        switch (dir) {
            case 0: next.x += 1; break; // Right
            case 1: next.x -= 1; break; // Left
            case 2: next.y += 1; break; // Up
            case 3: next.y -= 1; break; // Down
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
        double distance = sqrt(current.x * current.x + current.y * current.y);
        if (distance >= R_max) {
            break;
        }
    }

    return path;
}