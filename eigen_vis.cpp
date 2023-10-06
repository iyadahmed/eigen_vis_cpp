#include <chrono>
#include <complex>
#include <iostream>
#include <random>

#include <omp.h>

#define EIGEN_DONT_PARALLELIZE // Disable Eigen parallelization because we use OpenMP, as suggested by Eigen documentation

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <raylib.h>

using namespace std::complex_literals;

constexpr int WINDOW_WIDTH = 1280;
constexpr int WINDOW_HEIGHT = 800;
constexpr int NUM_SAMPLES = 1e4;

constexpr int TARGET_FPS = 60;

constexpr int FRAME_TIME_POS_X = 10;
constexpr int FRAME_TIME_POS_Y = 10;
constexpr int FRAME_TIME_FONT_SIZE = 20;

constexpr float VIEWPORT_RANGE_START = -5;
constexpr float VIEWPORT_RANGE_END = 5;

int main()
{
    // Note from Eigen's docs: "With Eigen 3.3, and a fully C++11 compliant compiler calling initParallel() is optional."
    Eigen::initParallel();

    std::random_device rng_device;
    std::default_random_engine rng_engine(rng_device());
    std::uniform_real_distribution<float> uniform_dist(-10, 10);

    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Eigen Visualizer");

    Image image = GenImageColor(WINDOW_WIDTH, WINDOW_HEIGHT, RAYWHITE);
    Texture2D texture = LoadTextureFromImage(image);

    SetTargetFPS(TARGET_FPS);
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        ImageClearBackground(&image, RAYWHITE);
        auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < NUM_SAMPLES; i++)
        {

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

            Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(A, false);
            for (const auto &v : ces.eigenvalues())
            {
                double x = v.real();
                double y = v.imag();
                auto xi = static_cast<size_t>(((x - VIEWPORT_RANGE_START) / (VIEWPORT_RANGE_END - VIEWPORT_RANGE_START)) * WINDOW_WIDTH);
                auto yi = static_cast<size_t>(((y - VIEWPORT_RANGE_START) / (VIEWPORT_RANGE_END - VIEWPORT_RANGE_START)) * WINDOW_HEIGHT);
                if (xi < WINDOW_WIDTH && yi < WINDOW_HEIGHT)
                {
                    ImageDrawPixel(&image, xi, WINDOW_HEIGHT - 1 - yi, BLACK);
                }
            }
        }
        UpdateTexture(texture, image.data);
        DrawTexture(texture, 0, 0, WHITE);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        DrawText(("Frame time: " + std::to_string(frame_time)).c_str(),
                 FRAME_TIME_POS_X, FRAME_TIME_POS_Y, FRAME_TIME_FONT_SIZE, BLUE);

        EndDrawing();
    }
    ExportImage(image, "eigen_vis.png");
    UnloadImage(image);
    UnloadTexture(texture);
    CloseWindow();
    return 0;
}
