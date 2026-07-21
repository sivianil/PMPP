#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

// Matrix multiplication function straightforward algorithm that works on CPU (host)
void MatrixMultiplication(float* M, float* N, float* P, int Width) {
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            float sum = 0;  // Initialize sum for dot product
            for (int k = 0; k < Width; k++) {
                float a = M[i * Width + k];  // Element from matrix M
                float b = N[k * Width + j];  // Element from matrix N
                sum += a * b;  // Compute dot product
            }
            P[i * Width + j] = sum; // store the result in the output matrix P
        }
    }
}

int main(void) {
    const int Width = 3;
    float M[9] = {1.0f, 2.0f, 3.0f,
                  4.0f, 5.0f, 6.0f,
                  7.0f, 8.0f, 9.0f};
    float N[9] = {9.0f, 8.0f, 7.0f,
                  6.0f, 5.0f, 4.0f,
                  3.0f, 2.0f, 1.0f};
    float P[9] = {0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f};

    MatrixMultiplication(M, N, P, Width);

    printf("Result matrix:\n");
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%.1f ", P[i * Width + j]);
        }
        printf("\n");
    }

    return 0;
}

