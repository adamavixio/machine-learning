const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

/// BLAS (Basic Linear Algebra Subprograms) wrapper for high-performance matrix operations.
///
/// On macOS, this uses Apple's Accelerate framework for 50-100 GFLOPS performance.
/// On other platforms, this would fall back to the hand-rolled SIMD implementation.
///
/// Expected speedup: 2x on matmul operations (20-50x raw BLAS vs hand-rolled)
/// Overall training speedup: ~1.8x (matmul is ~50% of training time)

// Feature flag: controlled by build.zig -Duse_blas=true
pub const USE_BLAS = build_options.use_blas;

// Import C BLAS interface (only on macOS with USE_BLAS)
pub const cblas = if (USE_BLAS and builtin.target.os.tag == .macos) @cImport({
    @cInclude("Accelerate/Accelerate.h");
}) else struct {};

/// High-performance matrix multiply using BLAS: C = A @ B
///
/// Computes C[M×N] = A[M×K] @ B[K×N] using optimized BLAS routines.
///
/// This is equivalent to cblas_sgemm with alpha=1.0, beta=0.0 (overwrite C).
///
/// Arguments:
///   C: Output matrix [M × N] (will be overwritten)
///   A: Left matrix [M × K]
///   B: Right matrix [K × N]
///   M: Number of rows in A and C
///   K: Number of columns in A, rows in B
///   N: Number of columns in B and C
pub fn matmul_blas(
    C: []f32,
    A: []const f32,
    B: []const f32,
    M: usize,
    K: usize,
    N: usize,
) void {
    // Feature gate: only available on macOS with USE_BLAS enabled
    if (!USE_BLAS or builtin.target.os.tag != .macos) {
        @compileError("BLAS not available on this platform or USE_BLAS not enabled");
    }

    // Validate buffer sizes
    std.debug.assert(C.len >= M * N);
    std.debug.assert(A.len >= M * K);
    std.debug.assert(B.len >= K * N);

    // Call BLAS: C[M×N] = 1.0 * A[M×K] @ B[K×N] + 0.0 * C
    //
    // Parameters:
    //   Order: CblasRowMajor (row-major layout, C convention)
    //   TransA: CblasNoTrans (A is not transposed)
    //   TransB: CblasNoTrans (B is not transposed)
    //   M: Number of rows in A and C
    //   N: Number of columns in B and C
    //   K: Number of columns in A, rows in B
    //   alpha: Scalar multiplier for A @ B (1.0)
    //   A: Pointer to A matrix data
    //   lda: Leading dimension of A (stride between rows) = K
    //   B: Pointer to B matrix data
    //   ldb: Leading dimension of B = N
    //   beta: Scalar multiplier for C (0.0 = overwrite)
    //   C: Pointer to C matrix data (output)
    //   ldc: Leading dimension of C = N
    if (USE_BLAS) {
        cblas.cblas_sgemm(
            cblas.CblasRowMajor, // Row-major layout
            cblas.CblasNoTrans, // A not transposed
            cblas.CblasNoTrans, // B not transposed
            @intCast(M), // M rows of A/C
            @intCast(N), // N cols of B/C
            @intCast(K), // K cols of A, rows of B
            1.0, // alpha = 1.0
            A.ptr, // A data
            @intCast(K), // Leading dimension of A
            B.ptr, // B data
            @intCast(N), // Leading dimension of B
            0.0, // beta = 0.0 (overwrite C)
            C.ptr, // C data (output)
            @intCast(N), // Leading dimension of C
        );
    }
}

/// Matrix multiply with A transposed using BLAS: C = A^T @ B
///
/// Computes C[K×N] = A^T[K×M] @ B[M×N] using optimized BLAS routines.
///
/// Arguments:
///   C: Output matrix [K × N] (will be overwritten)
///   A: Left matrix [M × K] (will be transposed to [K × M])
///   B: Right matrix [M × N]
///   M: Number of rows in A (before transpose), rows in B
///   K: Number of columns in A (before transpose), rows in output C
///   N: Number of columns in B and C
pub fn matmulTransposeA_blas(
    C: []f32,
    A: []const f32,
    B: []const f32,
    M: usize,
    K: usize,
    N: usize,
) void {
    if (!USE_BLAS or builtin.target.os.tag != .macos) {
        @compileError("BLAS not available on this platform or USE_BLAS not enabled");
    }

    // Validate buffer sizes
    std.debug.assert(C.len >= K * N);
    std.debug.assert(A.len >= M * K);
    std.debug.assert(B.len >= M * N);

    // Call BLAS: C[K×N] = 1.0 * A^T[K×M] @ B[M×N] + 0.0 * C
    //
    // cblas_sgemm with TransA = CblasTrans
    // M_op = K (rows of A^T)
    // N_op = N (cols of B)
    // K_op = M (cols of A^T = rows of A, rows of B)
    if (USE_BLAS) {
        cblas.cblas_sgemm(
            cblas.CblasRowMajor, // Row-major layout
            cblas.CblasTrans, // A is transposed
            cblas.CblasNoTrans, // B not transposed
            @intCast(K), // M_op: rows of A^T and C
            @intCast(N), // N_op: cols of B and C
            @intCast(M), // K_op: cols of A^T, rows of B
            1.0, // alpha = 1.0
            A.ptr, // A data (will be transposed)
            @intCast(K), // lda: leading dimension of A (NOT A^T)
            B.ptr, // B data
            @intCast(N), // ldb: leading dimension of B
            0.0, // beta = 0.0 (overwrite C)
            C.ptr, // C data (output)
            @intCast(N), // ldc: leading dimension of C
        );
    }
}

/// Matrix multiply with B transposed using BLAS: C = A @ B^T
///
/// Computes C[M×K] = A[M×N] @ B^T[N×K] using optimized BLAS routines.
///
/// Arguments:
///   C: Output matrix [M × K] (will be overwritten)
///   A: Left matrix [M × N]
///   B: Right matrix [K × N] (will be transposed to [N × K])
///   M: Number of rows in A and C
///   N: Number of columns in A, rows in B (before transpose)
///   K: Number of columns in B (before transpose), columns in C
pub fn matmulTransposeB_blas(
    C: []f32,
    A: []const f32,
    B: []const f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    if (!USE_BLAS or builtin.target.os.tag != .macos) {
        @compileError("BLAS not available on this platform or USE_BLAS not enabled");
    }

    // Validate buffer sizes
    std.debug.assert(C.len >= M * K);
    std.debug.assert(A.len >= M * N);
    std.debug.assert(B.len >= K * N);

    // Call BLAS: C[M×K] = 1.0 * A[M×N] @ B^T[N×K] + 0.0 * C
    //
    // cblas_sgemm with TransB = CblasTrans
    // M_op = M (rows of A)
    // N_op = K (cols of B^T)
    // K_op = N (cols of A = rows of B^T = rows of B)
    if (USE_BLAS) {
        cblas.cblas_sgemm(
            cblas.CblasRowMajor, // Row-major layout
            cblas.CblasNoTrans, // A not transposed
            cblas.CblasTrans, // B is transposed
            @intCast(M), // M_op: rows of A and C
            @intCast(K), // N_op: cols of B^T and C
            @intCast(N), // K_op: cols of A, rows of B^T
            1.0, // alpha = 1.0
            A.ptr, // A data
            @intCast(N), // lda: leading dimension of A
            B.ptr, // B data (will be transposed)
            @intCast(N), // ldb: leading dimension of B (NOT B^T)
            0.0, // beta = 0.0 (overwrite C)
            C.ptr, // C data (output)
            @intCast(K), // ldc: leading dimension of C
        );
    }
}

/// Check if BLAS is available on this platform at runtime.
pub fn isAvailable() bool {
    return USE_BLAS and builtin.target.os.tag == .macos;
}

// Tests are only compiled when BLAS is available
test "BLAS availability check" {
    // This test should pass on all platforms
    const available = isAvailable();
    if (builtin.target.os.tag == .macos and USE_BLAS) {
        try std.testing.expect(available);
    } else {
        try std.testing.expect(!available);
    }
}

// Integration test for matmul_blas (only runs when BLAS is available)
// Uncomment when USE_BLAS is enabled
//
// test "matmul_blas basic 2x2" {
//     if (!isAvailable()) return error.SkipZigTest;
//
//     const allocator = std.testing.allocator;
//
//     // Test: [2×2] @ [2×2] = [2×2]
//     // A = [[1, 2],    B = [[5, 6],    C = [[19, 22],
//     //      [3, 4]]         [7, 8]]         [43, 50]]
//
//     var A = try allocator.alloc(f32, 4);
//     defer allocator.free(A);
//     var B = try allocator.alloc(f32, 4);
//     defer allocator.free(B);
//     var C = try allocator.alloc(f32, 4);
//     defer allocator.free(C);
//
//     A[0] = 1.0; A[1] = 2.0;
//     A[2] = 3.0; A[3] = 4.0;
//
//     B[0] = 5.0; B[1] = 6.0;
//     B[2] = 7.0; B[3] = 8.0;
//
//     matmul_blas(C, A, B, 2, 2, 2);
//
//     try std.testing.expectApproxEqAbs(@as(f32, 19.0), C[0], 1e-5);
//     try std.testing.expectApproxEqAbs(@as(f32, 22.0), C[1], 1e-5);
//     try std.testing.expectApproxEqAbs(@as(f32, 43.0), C[2], 1e-5);
//     try std.testing.expectApproxEqAbs(@as(f32, 50.0), C[3], 1e-5);
// }
