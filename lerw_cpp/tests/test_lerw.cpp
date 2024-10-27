#include <gtest/gtest.h>
#include <unordered_set>
#include "utils.h"
#include "lerw_2d.h"
#include "lerw_3d.h"
#include "random_utils.h"
#include <cmath>
#include <random>

class LERWTest : public ::testing::Test {
protected:
    std::mt19937 rng;  // RNG for tests
    
    void SetUp() override {
        // Initialize RNG with a fixed seed for reproducible tests
        rng = std::mt19937(12345);
    }
};

// Test that paths are self-avoiding
TEST_F(LERWTest, PathIsSelfAvoiding2D) {
    double R = 20.0;
    auto path = simulate_LERW_2D(R, rng);
    
    std::unordered_set<Vec2D> visited;
    for (const auto& point : path) {
        // Each point should only appear once
        EXPECT_TRUE(visited.insert(point).second) 
            << "Point (" << point.x << "," << point.y << ") appears multiple times";
    }
}

TEST_F(LERWTest, PathIsSelfAvoiding3D) {
    double R = 20.0;
    auto path = simulate_LERW_3D(R, rng);
    
    std::unordered_set<Vec3D> visited;
    for (const auto& point : path) {
        EXPECT_TRUE(visited.insert(point).second)
            << "Point (" << point.x << "," << point.y << "," << point.z << ") appears multiple times";
    }
}

// Test that paths reach the target radius
TEST_F(LERWTest, PathReachesTargetRadius2D) {
    double R = 20.0;
    auto path = simulate_LERW_2D(R, rng);
    
    auto last_point = path.back();
    double final_distance = sqrt(last_point.x * last_point.x + last_point.y * last_point.y);
    EXPECT_GE(final_distance, R);
}

TEST_F(LERWTest, PathReachesTargetRadius3D) {
    double R = 20.0;
    auto path = simulate_LERW_3D(R, rng);
    
    auto last_point = path.back();
    double final_distance = sqrt(last_point.x * last_point.x + 
                               last_point.y * last_point.y + 
                               last_point.z * last_point.z);
    EXPECT_GE(final_distance, R);
}

// Test scaling behavior
TEST_F(LERWTest, PathLengthScaling2D) {
    std::vector<double> R_values = {10, 20, 40};
    std::vector<double> lengths;
    
    for (double R : R_values) {
        double avg_length = 0;
        int trials = 100;
        for (int i = 0; i < trials; i++) {
            auto path = simulate_LERW_2D(R, rng);
            avg_length += path.size();
        }
        avg_length /= trials;
        lengths.push_back(avg_length);
    }
    
    // Check that length scales approximately as R^D where D ≈ 1.25
    double ratio1 = lengths[1] / lengths[0];  // Should be approximately 2^1.25
    double ratio2 = lengths[2] / lengths[1];
    
    EXPECT_NEAR(ratio1, pow(2, 1.25), 0.2);
    EXPECT_NEAR(ratio2, pow(2, 1.25), 0.2);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}