#include <iostream>
#include <cmath>
#include <fstream>

bool isPrime(int n) {
    int count = 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            count++;
            break;
        }
    }

    return !(count > 0);
}

bool sum_of_two_squares(int n) {
    if(n == 1) {
        return true;
    }

    while(n % 2 == 0) {
        n /= 2;
    }

    for(int p = 3; p <= (int)sqrt(n); p += 4) {
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
    std::ofstream outFile("./valid_nums.txt");

    if (!outFile) {
        std::cerr << "Error creating file!" << std::endl;
    }

    int x = 1;
    int limit = (int)pow(2, 14);
    while(true) {
        double x_sqrt = sqrt(x);

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