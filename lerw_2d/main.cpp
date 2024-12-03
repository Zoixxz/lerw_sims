#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <oneapi/tbb/partitioner.h>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <atomic>

// Intel TBB Headers
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>

// External RNG Libraries
#include "pcg_random.hpp"

// High-Performance Hash Maps
#include "robin_hood.h"

struct AlphaData {
    std::vector<double> CDF;
};

struct RSquaredData {
    std::vector<std::pair<int, int>> possible;
};

struct rngWrapper {
    using rng_type = pcg32_fast;
    rng_type rng;

    rngWrapper() : rng(pcg_extras::seed_seq_from<std::random_device>()) {}

    // Generate a random double in [0, 1)
    inline double generate_double() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng);
    }

    // Generate a random size_t in [0, n)
    inline size_t generate_int(size_t n) {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        return dist(rng);
    }
};

constexpr int DIM = 2;
constexpr int ALPHA_START = 11; // these values are *10
constexpr int ALPHA_END = 20;
constexpr int ALPHA_JUMP = 1;
constexpr int L_START = 7; // these are log_2
constexpr int L_END = 13;

std::vector<AlphaData> ALPHA_INFO;
std::vector<RSquaredData> RSQUARED_INFO;
std::vector<int> VALID_NUMS;

constexpr uint64_t OFFSET = static_cast<uint64_t>(1) << 31;
inline uint64_t encode_pair(int x, int y) {
    uint64_t ux = static_cast<uint32_t>(x + OFFSET);
    uint64_t uy = static_cast<uint32_t>(y + OFFSET);
    return (ux << 32) | uy;
}

struct Position {
    int x;
    int y;
    uint64_t encoded;

    Position() : x(0), y(0), encoded(encode_pair(0, 0)) {}
    Position(int x, int y) : x(x), y(y), encoded(encode_pair(x, y)) {}

    bool operator!=(const Position& other) const {
        return encoded != other.encoded;
    }
};


void construct_ALPHA_INFO() {
    int num_alphas = (ALPHA_END - ALPHA_START) / ALPHA_JUMP + 1;
    ALPHA_INFO.resize(num_alphas);

    std::vector<int> alphas;
    for (int a = ALPHA_START; a <= ALPHA_END; a += ALPHA_JUMP) {
        alphas.push_back(a);
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, alphas.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                int a = alphas[i];
                std::vector<double> local_cdf(VALID_NUMS.size());

                double zeta_sum = 0.0;
                double exponent = (DIM + a / 10.0) / 2.0;

                // Compute probabilities
                for (size_t j = 0; j < VALID_NUMS.size(); ++j) {
                    double log_prob = -exponent * std::log(VALID_NUMS[j]);
                    double prob = std::exp(log_prob);
                    zeta_sum += prob;
                    local_cdf[j] = zeta_sum;
                }

                for(size_t j = 0; j < local_cdf.size(); j++) {
                    local_cdf[j] /= zeta_sum;
                }

                AlphaData& alpha_data = ALPHA_INFO[a - ALPHA_START];
                alpha_data.CDF = std::move(local_cdf);
            }
        }
    );
}

void construct_VALID_NUMS_POSSIBLE_POS() {
    // Read valid nums from file into vector
    std::ifstream inFile("./valid_nums.txt");

    if (!inFile) {
        std::cerr << "Error opening file!" << std::endl;
        exit(1);
    }

    int valid_num;
    while (inFile >> valid_num) {
        VALID_NUMS.push_back(valid_num);
    }
    inFile.close();

    int max_valid_num = *std::max_element(VALID_NUMS.begin(), VALID_NUMS.end());
    RSQUARED_INFO.resize(max_valid_num + 1);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, VALID_NUMS.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t idx = range.begin(); idx != range.end(); ++idx) {
                int current_r_squared = VALID_NUMS[idx];
                std::vector<std::pair<int, int>> local_possible;
                int r = static_cast<int>(std::sqrt(current_r_squared));

                for (int a = 0; a <= r; a++) {
                    int b_squared = current_r_squared - a * a;
                    int b = static_cast<int>(std::sqrt(b_squared));
                    if (b * b == b_squared) {
                        local_possible.emplace_back(a, b);
                        if (a != 0) local_possible.emplace_back(-a, b);
                        if (b != 0) local_possible.emplace_back(a, -b);
                        if (a != 0 && b != 0) local_possible.emplace_back(-a, -b);
                    }
                }

                RSQUARED_INFO[current_r_squared].possible = std::move(local_possible);
            }
        }
    );
}

inline int generate_random_r_squared(int alpha, rngWrapper& RNG) {
    double p = RNG.generate_double();

    AlphaData& ad = ALPHA_INFO[alpha - ALPHA_START];
    auto it = std::lower_bound(ad.CDF.begin(), ad.CDF.end(), p);

    if(it == ad.CDF.end()) return VALID_NUMS.back();
    else return VALID_NUMS[it - ad.CDF.begin()];
}

inline Position get_next_pos(const Position& currPos, int r_squared, rngWrapper& RNG) {
    const std::vector<std::pair<int, int>>& possible = RSQUARED_INFO[r_squared].possible;

    size_t idx = RNG.generate_int(possible.size());
    const std::pair<int, int>& jump = possible[idx];

    return Position(currPos.x + jump.first, currPos.y + jump.second);
}

int simulate(int L, int alpha, rngWrapper& RNG) {
    std::vector<Position> path;
    path.reserve(L*L);
    robin_hood::unordered_flat_set<uint64_t> visited;

    Position curr_pos(0, 0);
    path.emplace_back(curr_pos);
    visited.insert(curr_pos.encoded);

    while (true) {
        int r_squared = generate_random_r_squared(alpha, RNG);

        Position next_pos = get_next_pos(curr_pos, r_squared, RNG);

        auto [it, inserted] = visited.insert(next_pos.encoded);
        if(!inserted) {
            // loop formed
            while(path.back() != next_pos) {
                visited.erase(path.back().encoded);
                path.pop_back();
            }
        } else {
            // New position
            path.emplace_back(next_pos);
        }

        curr_pos = next_pos;

        int dist_squared = curr_pos.x * curr_pos.x + curr_pos.y * curr_pos.y;
        if (dist_squared >= L * L) break;
    }

    return static_cast<int>(path.size());
}

long double simulate_lr_2d(int L, int alpha, int num_trials) {
    tbb::combinable<long double> total_length([]() { return 0.0L; });

    tbb::parallel_for(tbb::blocked_range<int>(0, num_trials),
        [&](const tbb::blocked_range<int>& range) {
            rngWrapper rng;
            for (int i = range.begin(); i != range.end(); ++i) {
                int length = simulate(L, alpha, rng);
                total_length.local() += length;
            }
        },
        tbb::auto_partitioner()
    );

    return total_length.combine(std::plus<long double>()) / num_trials;
}

int main() {
    // Precompute valid nums and possible positions
    std::cout << "Constructing valid nums and possible positions..." << std::endl;
    construct_VALID_NUMS_POSSIBLE_POS();
    std::cout << "Done." << std::endl;

    std::cout << "Constructing ALPHA_INFO..." << std::endl;
    construct_ALPHA_INFO();
    std::cout << "Done." << std::endl;

    std::ofstream outFile("./results.txt", std::ios_base::app);

    if (!outFile) {
        std::cerr << "Error creating file!" << std::endl;
        exit(1);
    }

    outFile.precision(15);
    outFile << std::fixed;

    const int num_trials = static_cast<int>(std::pow(10, 6));

    for (int alpha = ALPHA_START; alpha <= ALPHA_END; alpha += ALPHA_JUMP) {
        std::cout << "Processing alpha = " << alpha << "..." << std::endl;

        for (int i = L_START; i <= L_END; i++) {
            int L = 1 << i; // Use bit shifting for power of 2

            auto start = std::chrono::high_resolution_clock::now();

            long double avg_length = simulate_lr_2d(L, alpha, num_trials);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            outFile << alpha << " " << i << " " << avg_length << std::endl;
            std::cout << "L = " << L << ", Average path length: " << avg_length << ", time: " << elapsed.count() << " seconds" << std::endl;
        }
    }
}
