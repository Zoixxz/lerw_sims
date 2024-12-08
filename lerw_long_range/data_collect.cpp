#include <iostream>
#include <cmath>
#include <fstream>

bool isPrime(long long n) {
    int count = 0;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            count++;
            break;
        }
    }

    return !(count > 0);
}

bool sum_of_two_squares(long long n) {
    if(n == 1) {
        return true;
    }

    while(n % 2 == 0) {
        n /= 2;
    }

    for(long long p = 3; p <= (int)sqrt(n); p += 4) {
        if(isPrime(p)) {
            if(n % p == 0) {
                int count = 0;
                while(n % p == 0) {
                    n /= p;
                    count++; 
                }
                if(count % 2 != 0) {
                    return false;
                }
            }
        }
    }

    if(n > 1 && n % 4 == 3) {
        return false;
    }

    return true;
}

int main() {
    std::ofstream outFile("/home/epsilon/Code/Research/Long_Range_Simulations/lerw_sims/lerw_2d/valid_nums.txt", std::ios_base::app);

    if (!outFile) {
        std::cerr << "Error creating file!" << std::endl;
    }

    long long x = std::pow(2, 28) + 1;
    long long limit = std::pow(2, 15);
    while(true) {
        long double x_sqrt = std::sqrt(x);

        if(x % 1000000 == 0) {
            std::cout << x_sqrt << std::endl;
        }

        if(sum_of_two_squares(x)) {
            outFile << x << std::endl;
        }

        if(x_sqrt >= limit) {
            std::cout << x << std::endl;
            break;
        }

        x++;
    }
    outFile.close();
}