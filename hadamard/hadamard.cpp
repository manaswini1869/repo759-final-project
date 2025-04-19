#include <iostream>
#include <vector>

using namespace std;

// Assume square matrices
using Matrix = vector<vector<double>>;

// Hadamard Transform (simple, not optimized)
void hadamardTransform(Matrix& mat, bool inverse = false) {
    int n = mat.size();
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < len; ++k) {
                    double u = mat[i + k][j];
                    double v = mat[i + k + len][j];
                    mat[i + k][j] = u + v;
                    mat[i + k + len][j] = u - v;
                }
            }
        }
    }
}

// Transpose of a matrix
Matrix transpose(const Matrix& mat) {
    int n = mat.size();
    Matrix res(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            res[j][i] = mat[i][j];
    return res;
}

// Matrix multiplication
Matrix multiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix res(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                res[i][j] += A[i][k] * B[k][j];
    return res;
}

// Main calculation: deltaW = H * C * H^T
Matrix computeDeltaW(const Matrix& C) {
    int n = C.size();
    
    // 1. Create Hadamard matrix H (basic version)
    Matrix H(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        H[i][i] = 1; // Identity for simplicity; you can build true Hadamard if needed
    
    hadamardTransform(H); // Transform it into Hadamard basis
    
    Matrix H_T = transpose(H); // H transpose

    // 2. deltaW = H * C * H^T
    Matrix temp = multiply(H, C);
    Matrix deltaW = multiply(temp, H_T);

    return deltaW;
}

int main() {
    // Example: C is a 4x4 matrix
    Matrix C = {
        {1, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 3, 0},
        {0, 0, 0, 4}
    };

    Matrix deltaW = computeDeltaW(C);

    cout << "Delta W:\n";
    for (auto& row : deltaW) {
        for (double val : row)
            cout << val << " ";
        cout << endl;
    }

    return 0;
}
