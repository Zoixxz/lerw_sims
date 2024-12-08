#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <oneapi/tbb/partitioner.h>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <gmp.h>
#include <gmpxx.h>
#include <mpfr.h>

// Intel TBB Headers
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>

// External RNG Libraries
#include "pcg_random.hpp"

// High-Performance Hash Maps
#include "robin_hood.h"

constexpr int PREC = 160;

struct CDF_DATA {
    std::vector<mpf_class> CDF;
};

struct rngWrapper {
    using rng_type = pcg32_fast;
    rng_type rng;

    rngWrapper(uint64_t seed) : rng(seed) {}

    inline mpf_class generate_high_precision_float(int bits = PREC) {
        mpf_class result(0.0, PREC);
        mpf_class base(1.0 / 4294967296.0, PREC);

        int bits_generated = 0;

        while (bits_generated < bits) {
            uint32_t rand32 = rng(); // Generate 32 random bits
            result += mpf_class(rand32, PREC) * base;
            base /= 4294967296.0; // Divide by 2^32
            bits_generated += 32;
        }

        return result;
    }

    // Generate a random size_t in [0, n)
    inline size_t generate_int(size_t n) {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        return dist(rng);
    }
};

constexpr int DIM = 2;
constexpr int ALPHA_START = 2; // these values are *10
constexpr int ALPHA_END = 2;
constexpr int ALPHA_JUMP = 1;
constexpr int L_START = 15; // these are log_2
constexpr int L_END = 18;

std::vector<CDF_DATA> LENGTH_INFO;

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


void construct_LENGTH_INFO(int alpha) {
    int num_lengths = L_END - L_START + 1;
    LENGTH_INFO.resize(num_lengths);

    double exponent = DIM + alpha / 10.0 - 1.0;

    std::vector<int> lengths;
    for (int L = L_START; L <= L_END; L++) {
        lengths.push_back(L);
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, lengths.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                int L = 1 << lengths[i];
                std::vector<mpf_class> local_cdf(2*L);

                mpf_class zeta_sum(0.0, PREC);

                mpfr_t log_r, log_prob, prob;
                mpfr_init2(log_r, PREC);
                mpfr_init2(log_prob, PREC);
                mpfr_init2(prob, PREC);

                // Compute probabilities
                for (int r = 1; r <= 2*L; r++) {
                    mpfr_set_ui(log_r, r, MPFR_RNDN);
                    mpfr_log(log_r, log_r, MPFR_RNDN);
                    mpfr_mul_d(log_prob, log_r, exponent, MPFR_RNDN);
                    mpfr_neg(log_prob, log_prob, MPFR_RNDN);
                    mpfr_exp(prob, log_prob, MPFR_RNDN);
                    mpfr_mul_d(prob, prob, 8.0, MPFR_RNDN);

                    mpf_t prob_as_mpf;
                    mpf_init2(prob_as_mpf, PREC);
                    mpfr_get_f(prob_as_mpf, prob, MPFR_RNDN);
                    zeta_sum += mpf_class(prob_as_mpf, PREC);
                    mpf_clear(prob_as_mpf);

                    local_cdf[r - 1] = zeta_sum;
                }

                // Normalize CDF
                for (size_t j = 0; j < local_cdf.size(); j++) {
                    local_cdf[j] /= zeta_sum;
                }

                CDF_DATA& cdf_data = LENGTH_INFO[lengths[i] - L_START];
                cdf_data.CDF = std::move(local_cdf);

                mpfr_clear(log_r);
                mpfr_clear(log_prob);
                mpfr_clear(prob);
            }
        },
        tbb::auto_partitioner()
    );
}

inline int generate_random_r(int L, rngWrapper& RNG) {
    mpf_class p = RNG.generate_high_precision_float();

    CDF_DATA& cd = LENGTH_INFO[L - L_START];
    
    auto it = std::lower_bound(cd.CDF.begin(), cd.CDF.end(), p,
    [](const mpf_class& cdf_val, const mpf_class& target) {
            return cdf_val < target;
        });

    return it - cd.CDF.begin() + 1;
}

inline Position get_next_pos(const Position& currPos, int r, rngWrapper& RNG) {
    // Total number of possible vectors with L-infty norm r in 2D
    size_t total_vectors = 8 * r;

    // Generate a random index in [0, total_vectors - 1]
    size_t k = RNG.generate_int(total_vectors);

    int x, y;
    // Determine which quadrant or axis the vector falls into
    if (k < 2 * (2 * r + 1)) {
        // Case 1: x = r or x = -r
        // Each has (2r + 1) possible y values from -r to r
        int sign = (k < (2 * r + 1)) ? 1 : -1;
        y = -r + static_cast<int>(k % (2 * r + 1));
        x = sign * r;
    }
    else {
        // Case 2: y = r or y = -r
        // Each has (2r - 1) possible x values from -r + 1 to r - 1
        k -= 2 * (2 * r + 1);
        int sign = (k < (2 * r - 1)) ? 1 : -1;
        x = -r + 1 + static_cast<int>(k % (2 * r - 1));
        y = sign * r;
    }

    return Position(currPos.x + x, currPos.y + y);
}

int simulate(int L, rngWrapper& RNG) {
    std::vector<Position> path;
    robin_hood::unordered_flat_set<uint64_t> visited;

    int L_real = 1 << L;

    Position curr_pos(0, 0);
    path.emplace_back(curr_pos);
    visited.insert(curr_pos.encoded);

    while (true) {
        int r = generate_random_r(L, RNG);

        Position next_pos = get_next_pos(curr_pos, r, RNG);

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

        if(std::max(std::abs(curr_pos.x), std::abs(curr_pos.y)) >= L_real) break;
    }

    return static_cast<int>(path.size());
}

long double simulate_lr_2d(int L, int num_trials) {
    tbb::combinable<long double> total_length([]() { return 0.0L; });

    tbb::parallel_for(tbb::blocked_range<int>(0, num_trials),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                uint64_t seed = static_cast<uint64_t>(i) + 1;
                rngWrapper rng(seed);
                int length = simulate(L, rng);
                total_length.local() += length;
            }
        },
        tbb::auto_partitioner()
    );

    return total_length.combine(std::plus<long double>()) / num_trials;
}

int main() {
    mpf_set_default_prec(PREC);

    std::ofstream outFile("./results_2d_1.txt", std::ios_base::app);

    if (!outFile) {
        std::cerr << "Error creating file!" << std::endl;
        exit(1);
    }

    outFile.precision(15);
    outFile << std::fixed;

    const int num_trials = static_cast<int>(std::pow(10, 5));

    for (int alpha = ALPHA_START; alpha <= ALPHA_END; alpha += ALPHA_JUMP) {
        std::cout << "Processing alpha = " << alpha << "..." << std::endl;

        std::cout << "Constructing length info..." << std::endl;
        LENGTH_INFO.clear();
        construct_LENGTH_INFO(alpha);
        std::cout << "Done." << std::endl;

        for (int L = L_START; L <= L_END; L++) {
            auto start = std::chrono::high_resolution_clock::now();

            long double avg_length = simulate_lr_2d(L, num_trials);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            outFile << alpha << " " << L << " " << avg_length << std::endl;
            std::cout << "L = " << (1 << L) << ", Average path length: " << avg_length << ", time: " << elapsed.count() << " seconds" << std::endl;
        }
    }
}