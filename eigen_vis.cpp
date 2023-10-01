#include <complex>
#include <cstddef> // for size_t
#include <cstring> // for memset
#include <fstream>
#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std::complex_literals;

constexpr size_t N = 4096 * 2;

int main()
{
    Eigen::MatrixXi output(N, N);
    output.fill(0);
    {
        std::random_device rng_device;
        std::default_random_engine rng_engine(rng_device());
        std::uniform_real_distribution<float> uniform_dist(-10, 10);

        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
        std::complex<float> i = -1if;
        for (int j = 0; j < 1e6; j++)
        {
            float t1 = uniform_dist(rng_engine);
            float t2 = uniform_dist(rng_engine);

            Eigen::MatrixXcd A(7, 7);
            A(0, 0) = -i;
            A(0, 1) = -i;
            A(0, 2) = i;
            A(0, 3) = 0;
            A(0, 4) = -i;
            A(0, 5) = 0;
            A(0, 6) = -i;

            A(1, 0) = -i;
            A(1, 1) = 0;
            A(1, 2) = 0.3;
            A(1, 3) = -i;
            A(1, 4) = 0;
            A(1, 5) = 0;
            A(1, 6) = -i;

            A(2, 0) = 0;
            A(2, 1) = -i;
            A(2, 2) = 0;
            A(2, 3) = 0.3;
            A(2, 4) = i;
            A(2, 5) = 0.3;
            A(2, 6) = i;

            A(3, 0) = 0;
            A(3, 1) = -i;
            A(3, 2) = i;
            A(3, 3) = -i;
            A(3, 4) = i;
            A(3, 5) = 0.3;
            A(3, 6) = -i;

            A(4, 0) = 0;
            A(4, 1) = i;
            A(4, 2) = i;
            A(4, 3) = 0;
            A(4, 4) = 0;
            A(4, 5) = -i;
            A(4, 6) = i;

            A(5, 0) = 0;
            A(5, 1) = 0.3;
            A(5, 2) = 0;
            A(5, 3) = -i;
            A(5, 4) = 0;
            A(5, 5) = t1;
            A(5, 6) = 0;

            A(6, 0) = -i;
            A(6, 1) = t2;
            A(6, 2) = i;
            A(6, 3) = i;
            A(6, 4) = i;
            A(6, 5) = 0.3;
            A(6, 6) = -i;

            ces.compute(A, false);

            for (const auto &v : ces.eigenvalues())
            {
                double x = v.real();
                double y = v.imag();
                auto xi = static_cast<size_t>(((x + 5) / 10) * N);
                auto yi = static_cast<size_t>(((y + 5) / 10) * N);
                if (xi < N && yi < N)
                    output(xi, yi) = 1;
            }
        }
    }

    std::ofstream ofs("output.pbm", std::ofstream::binary);
    ofs << "P1\n"
        << N << ' ' << N << '\n';
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            ofs << output(i, j) << ' ';
        }
        ofs << '\n';
    }
    std::cout << "Finished" << std::endl;
    return 0;
}
