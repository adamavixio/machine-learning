const std = @import("std");
const config = @import("config.zig");
const blas = @import("blas.zig");

const Scalar = config.TensorConfig.Scalar;
const Vec = config.Vec;
const simd_lanes = config.TensorConfig.simd_lanes;

/// Naive matrix multiplication: C = A @ B
/// A: [M x K], B: [K x N], C: [M x N]
/// Row-major layout assumed
pub fn matmulNaive(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    K: usize,
    N: usize,
) !void {
    if (C.len != M * N or A.len != M * K or B.len != K * N) {
        return error.ShapeMismatch;
    }

    // Zero output
    @memset(C, 0.0);

    // C[i,j] = sum_k A[i,k] * B[k,j]
    for (0..M) |i| {
        for (0..K) |k| {
            const a_val = A[i * K + k];
            const b_row_start = k * N;
            const c_row_start = i * N;

            // SIMD over N dimension
            const vec_count = N / simd_lanes;
            const remainder = N % simd_lanes;

            const a_vec: Vec = @splat(a_val);

            for (0..vec_count) |v| {
                const offset = v * simd_lanes;
                const b_vec: Vec = B[b_row_start + offset ..][0..simd_lanes].*;
                var c_vec: Vec = C[c_row_start + offset ..][0..simd_lanes].*;
                c_vec += a_vec * b_vec;
                const c_array: [simd_lanes]Scalar = c_vec;
                @memcpy(C[c_row_start + offset ..][0..simd_lanes], &c_array);
            }

            // Scalar tail
            const tail_offset = vec_count * simd_lanes;
            for (0..remainder) |j| {
                C[c_row_start + tail_offset + j] += a_val * B[b_row_start + tail_offset + j];
            }
        }
    }
}

/// Blocked matrix multiplication for better cache locality
/// Block size tuned for typical L1 cache
pub fn matmulBlocked(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    K: usize,
    N: usize,
) !void {
    if (C.len != M * N or A.len != M * K or B.len != K * N) {
        return error.ShapeMismatch;
    }

    const block_size = 32; // Tuned for cache

    // Zero output
    @memset(C, 0.0);

    // Blocked outer loops
    var ii: usize = 0;
    while (ii < M) : (ii += block_size) {
        var jj: usize = 0;
        while (jj < N) : (jj += block_size) {
            var kk: usize = 0;
            while (kk < K) : (kk += block_size) {
                // Inner block
                const i_end = @min(ii + block_size, M);
                const j_end = @min(jj + block_size, N);
                const k_end = @min(kk + block_size, K);

                for (ii..i_end) |i| {
                    for (kk..k_end) |k| {
                        const a_val = A[i * K + k];
                        const a_vec: Vec = @splat(a_val);
                        const b_row_start = k * N + jj;
                        const c_row_start = i * N + jj;
                        const block_width = j_end - jj;

                        const vec_count = block_width / simd_lanes;
                        const remainder = block_width % simd_lanes;

                        // SIMD
                        for (0..vec_count) |v| {
                            const offset = v * simd_lanes;
                            const b_vec: Vec = B[b_row_start + offset ..][0..simd_lanes].*;
                            var c_vec: Vec = C[c_row_start + offset ..][0..simd_lanes].*;
                            c_vec += a_vec * b_vec;
                            const c_array: [simd_lanes]Scalar = c_vec;
                            @memcpy(C[c_row_start + offset ..][0..simd_lanes], &c_array);
                        }

                        // Scalar tail
                        const tail_offset = vec_count * simd_lanes;
                        for (0..remainder) |j| {
                            C[c_row_start + tail_offset + j] += a_val * B[b_row_start + tail_offset + j];
                        }
                    }
                }
            }
        }
    }
}

/// Public API - chooses best implementation
pub fn matmul(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    K: usize,
    N: usize,
) !void {
    // One-time diagnostic to verify BLAS path
    const DebugOnce = struct {
        var printed = false;
    };
    if (!DebugOnce.printed) {
        if (comptime blas.isAvailable()) {
            std.debug.print("[MATMUL] Using Apple Accelerate BLAS (compiled with -Duse_blas=true)\n", .{});
        } else {
            std.debug.print("[MATMUL] BLAS not available, using hand-rolled SIMD fallback\n", .{});
        }
        DebugOnce.printed = true;
    }

    // Compile-time dispatch to BLAS if available (2-10x speedup on matmul)
    if (comptime blas.isAvailable()) {
        blas.matmul_blas(C, A, B, M, K, N);
        return;
    }

    // Otherwise use blocked version for larger matrices
    if (M >= 64 and N >= 64 and K >= 64) {
        return matmulBlocked(C, A, B, M, K, N);
    } else {
        return matmulNaive(C, A, B, M, K, N);
    }
}

test "matmul small" {
    const allocator = std.testing.allocator;

    // 2x3 @ 3x2 = 2x2
    const A = try allocator.alloc(Scalar, 6);
    defer allocator.free(A);
    const B = try allocator.alloc(Scalar, 6);
    defer allocator.free(B);
    const C = try allocator.alloc(Scalar, 4);
    defer allocator.free(C);

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    A[0] = 1.0;
    A[1] = 2.0;
    A[2] = 3.0;
    A[3] = 4.0;
    A[4] = 5.0;
    A[5] = 6.0;

    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6]]
    B[0] = 1.0;
    B[1] = 2.0;
    B[2] = 3.0;
    B[3] = 4.0;
    B[4] = 5.0;
    B[5] = 6.0;

    try matmul(C, A, B, 2, 3, 2);

    // C = [[22, 28],
    //      [49, 64]]
    try std.testing.expectApproxEqAbs(@as(Scalar, 22.0), C[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 28.0), C[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 49.0), C[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 64.0), C[3], 1e-5);
}

test "matmul identity" {
    const allocator = std.testing.allocator;
    const n = 4;

    const A = try allocator.alloc(Scalar, n * n);
    defer allocator.free(A);
    const I = try allocator.alloc(Scalar, n * n);
    defer allocator.free(I);
    const C = try allocator.alloc(Scalar, n * n);
    defer allocator.free(C);

    // Random matrix A
    for (A, 0..) |*val, i| val.* = @floatFromInt(i);

    // Identity matrix I
    @memset(I, 0.0);
    for (0..n) |i| I[i * n + i] = 1.0;

    try matmul(C, A, I, n, n, n);

    // A @ I should equal A
    for (A, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, C[i], 1e-5);
    }
}

/// Naive implementation of C = A @ B^T (B is transposed)
/// A: [M x N], B: [K x N], C: [M x K]
fn matmulTransposeBNaive(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    N: usize,
    K: usize,
) !void {
    @memset(C, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            for (0..N) |n| {
                // A[i,n] * B[k,n] (B is transposed)
                C[i * K + k] += A[i * N + n] * B[k * N + n];
            }
        }
    }
}

/// Naive implementation of C = A^T @ B (A is transposed)
/// A: [M x K], B: [M x N], C: [K x N]
fn matmulTransposeANaive(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    K: usize,
    N: usize,
) !void {
    @memset(C, 0.0);
    for (0..K) |k| {
        for (0..N) |n| {
            for (0..M) |m| {
                // A[m,k] transposed = A^T[k,m], B[m,n]
                C[k * N + n] += A[m * K + k] * B[m * N + n];
            }
        }
    }
}

/// Public API: Matrix multiply with B transposed: C = A @ B^T
/// A: [M x N], B: [K x N], C: [M x K]
pub fn matmulTransposeB(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    N: usize,
    K: usize,
) !void {
    // Compile-time dispatch to BLAS if available
    if (comptime blas.isAvailable()) {
        blas.matmulTransposeB_blas(C, A, B, M, N, K);
        return;
    }

    // Fallback to naive implementation
    return matmulTransposeBNaive(C, A, B, M, N, K);
}

/// Public API: Matrix multiply with A transposed: C = A^T @ B
/// A: [M x K], B: [M x N], C: [K x N]
pub fn matmulTransposeA(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    K: usize,
    N: usize,
) !void {
    // Compile-time dispatch to BLAS if available
    if (comptime blas.isAvailable()) {
        blas.matmulTransposeA_blas(C, A, B, M, K, N);
        return;
    }

    // Fallback to naive implementation
    return matmulTransposeANaive(C, A, B, M, K, N);
}
