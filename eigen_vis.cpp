#include <chrono>
#include <complex>
#include <cstddef> // for size_t
#include <cstdint>
#include <cstring> // for memset
#include <fstream>
#include <iostream>
#include <random>

#include <omp.h>

#define EIGEN_DONT_PARALLELIZE // Disable Eigen parallelization because we use OpenMP, as suggested by Eigen documentation

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std::complex_literals;

constexpr size_t N = 4096 * 2;

int main()
{
    // Note from docs: "With Eigen 3.3, and a fully C++11 compliant compiler calling initParallel() is optional."
    Eigen::initParallel();

    Eigen::Matrix<uint8_t, -1, -1> output(N, N);
    output.fill(255);
    std::random_device rng_device;
    std::default_random_engine rng_engine(rng_device());
    std::uniform_real_distribution<float> uniform_dist(-10, 10);

    constexpr int num_samples = 1e6;

    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < num_samples; i++)
    {
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
        float t1 = uniform_dist(rng_engine);
        float t2 = uniform_dist(rng_engine);

        Eigen::MatrixXcd A(7, 7);
        A(0, 0) = -1i;
        A(0, 1) = -1i;
        A(0, 2) = 1i;
        A(0, 3) = 0;
        A(0, 4) = -1i;
        A(0, 5) = 0;
        A(0, 6) = -1i;

        A(1, 0) = -1i;
        A(1, 1) = 0;
        A(1, 2) = 0.3;
        A(1, 3) = -1i;
        A(1, 4) = 0;
        A(1, 5) = 0;
        A(1, 6) = -1i;

        A(2, 0) = 0;
        A(2, 1) = -1i;
        A(2, 2) = 0;
        A(2, 3) = 0.3;
        A(2, 4) = 1i;
        A(2, 5) = 0.3;
        A(2, 6) = 1i;

        A(3, 0) = 0;
        A(3, 1) = -1i;
        A(3, 2) = 1i;
        A(3, 3) = -1i;
        A(3, 4) = 1i;
        A(3, 5) = 0.3;
        A(3, 6) = -1i;

        A(4, 0) = 0;
        A(4, 1) = 1i;
        A(4, 2) = 1i;
        A(4, 3) = 0;
        A(4, 4) = 0;
        A(4, 5) = -1i;
        A(4, 6) = 1i;

        A(5, 0) = 0;
        A(5, 1) = 0.3;
        A(5, 2) = 0;
        A(5, 3) = -1i;
        A(5, 4) = 0;
        A(5, 5) = t1;
        A(5, 6) = 0;

        A(6, 0) = -1i;
        A(6, 1) = t2;
        A(6, 2) = 1i;
        A(6, 3) = 1i;
        A(6, 4) = 1i;
        A(6, 5) = 0.3;
        A(6, 6) = -1i;

        ces.compute(A, false);
        for (const auto &v : ces.eigenvalues())
        {
            double x = v.real();
            double y = v.imag();
            auto xi = static_cast<size_t>(((x + 5) / 10) * N);
            auto yi = static_cast<size_t>(((y + 5) / 10) * N);
            if (xi < N && yi < N)
                output(xi, yi) = 0;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;

    std::ofstream ofs("output.pgm", std::ofstream::binary);
    ofs << "P5\n"
        << N << ' ' << N << "\n255\n";
    // Otherwise the image will be flipped due to how PGM file format is layed out
    output.rowwise().reverseInPlace();
    ofs.write(reinterpret_cast<char *>(output.data()), N * N * sizeof(uint8_t));
    std::cout << "Finished" << std::endl;
    return 0;
}
