const std = @import("std");
const config = @import("config.zig");
const tensor_mod = @import("tensor.zig");

const Scalar = config.TensorConfig.Scalar;
const Vec = config.Vec;
const Tensor = tensor_mod.Tensor;
const simd_lanes = config.TensorConfig.simd_lanes;

/// Add two tensors element-wise: out = a + b
pub fn addInto(out: []Scalar, a: []const Scalar, b: []const Scalar) !void {
    if (out.len != a.len or out.len != b.len) {
        return error.ShapeMismatch;
    }

    const n = out.len;
    const vec_count = n / simd_lanes;
    const remainder = n % simd_lanes;

    // SIMD path
    var i: usize = 0;
    while (i < vec_count) : (i += 1) {
        const offset = i * simd_lanes;
        const a_vec: Vec = a[offset..][0..simd_lanes].*;
        const b_vec: Vec = b[offset..][0..simd_lanes].*;
        const result_vec = a_vec + b_vec;
        const result_array: [simd_lanes]Scalar = result_vec;
        @memcpy(out[offset..][0..simd_lanes], &result_array);
    }

    // Scalar tail
    const tail_offset = vec_count * simd_lanes;
    for (0..remainder) |j| {
        out[tail_offset + j] = a[tail_offset + j] + b[tail_offset + j];
    }
}

/// Multiply two tensors element-wise: out = a * b
pub fn mulInto(out: []Scalar, a: []const Scalar, b: []const Scalar) !void {
    if (out.len != a.len or out.len != b.len) {
        return error.ShapeMismatch;
    }

    const n = out.len;
    const vec_count = n / simd_lanes;
    const remainder = n % simd_lanes;

    // SIMD path
    var i: usize = 0;
    while (i < vec_count) : (i += 1) {
        const offset = i * simd_lanes;
        const a_vec: Vec = a[offset..][0..simd_lanes].*;
        const b_vec: Vec = b[offset..][0..simd_lanes].*;
        const result_vec = a_vec * b_vec;
        const result_array: [simd_lanes]Scalar = result_vec;
        @memcpy(out[offset..][0..simd_lanes], &result_array);
    }

    // Scalar tail
    const tail_offset = vec_count * simd_lanes;
    for (0..remainder) |j| {
        out[tail_offset + j] = a[tail_offset + j] * b[tail_offset + j];
    }
}

/// ReLU activation: out = max(0, x)
pub fn reluInto(out: []Scalar, x: []const Scalar) !void {
    if (out.len != x.len) {
        return error.ShapeMismatch;
    }

    const n = out.len;
    const vec_count = n / simd_lanes;
    const remainder = n % simd_lanes;

    const zero_vec: Vec = @splat(0.0);

    // SIMD path
    var i: usize = 0;
    while (i < vec_count) : (i += 1) {
        const offset = i * simd_lanes;
        const x_vec: Vec = x[offset..][0..simd_lanes].*;
        const result_vec = @select(Scalar, x_vec > zero_vec, x_vec, zero_vec);
        const result_array: [simd_lanes]Scalar = result_vec;
        @memcpy(out[offset..][0..simd_lanes], &result_array);
    }

    // Scalar tail
    const tail_offset = vec_count * simd_lanes;
    for (0..remainder) |j| {
        const val = x[tail_offset + j];
        out[tail_offset + j] = if (val > 0.0) val else 0.0;
    }
}

/// Tanh activation (scalar fallback - @tanh not vectorizable)
pub fn tanhInto(out: []Scalar, x: []const Scalar) !void {
    if (out.len != x.len) {
        return error.ShapeMismatch;
    }

    for (x, 0..) |val, i| {
        out[i] = std.math.tanh(val);
    }
}

/// Sum reduction: returns sum of all elements
pub fn sum(x: []const Scalar) Scalar {
    const n = x.len;
    const vec_count = n / simd_lanes;
    const remainder = n % simd_lanes;

    var sum_vec: Vec = @splat(0.0);

    // SIMD path
    var i: usize = 0;
    while (i < vec_count) : (i += 1) {
        const offset = i * simd_lanes;
        const x_vec: Vec = x[offset..][0..simd_lanes].*;
        sum_vec += x_vec;
    }

    // Reduce vector to scalar
    var total = @reduce(.Add, sum_vec);

    // Scalar tail
    const tail_offset = vec_count * simd_lanes;
    for (0..remainder) |j| {
        total += x[tail_offset + j];
    }

    return total;
}

/// Max reduction: returns maximum element
pub fn max(x: []const Scalar) Scalar {
    if (x.len == 0) return -std.math.inf(Scalar);

    const n = x.len;
    const vec_count = n / simd_lanes;
    const remainder = n % simd_lanes;

    var max_vec: Vec = @splat(-std.math.inf(Scalar));

    // SIMD path
    var i: usize = 0;
    while (i < vec_count) : (i += 1) {
        const offset = i * simd_lanes;
        const x_vec: Vec = x[offset..][0..simd_lanes].*;
        max_vec = @select(Scalar, x_vec > max_vec, x_vec, max_vec);
    }

    // Reduce vector to scalar
    var max_val = @reduce(.Max, max_vec);

    // Scalar tail
    const tail_offset = vec_count * simd_lanes;
    for (0..remainder) |j| {
        const val = x[tail_offset + j];
        if (val > max_val) max_val = val;
    }

    return max_val;
}

test "elementwise add" {
    const allocator = std.testing.allocator;
    const n = 100;

    const a = try allocator.alloc(Scalar, n);
    defer allocator.free(a);
    const b = try allocator.alloc(Scalar, n);
    defer allocator.free(b);
    const out = try allocator.alloc(Scalar, n);
    defer allocator.free(out);

    for (a, 0..) |*val, i| val.* = @floatFromInt(i);
    for (b, 0..) |*val, i| val.* = @floatFromInt(i * 2);

    try addInto(out, a, b);

    for (out, 0..) |val, i| {
        const expected = @as(Scalar, @floatFromInt(i * 3));
        try std.testing.expectApproxEqAbs(expected, val, 1e-5);
    }
}

test "relu activation" {
    const allocator = std.testing.allocator;
    const n = 10;

    const x = try allocator.alloc(Scalar, n);
    defer allocator.free(x);
    const out = try allocator.alloc(Scalar, n);
    defer allocator.free(out);

    for (x, 0..) |*val, i| {
        val.* = @floatFromInt(@as(i32, @intCast(i)) - 5);
    }

    try reluInto(out, x);

    try std.testing.expectEqual(@as(Scalar, 0.0), out[0]);
    try std.testing.expectEqual(@as(Scalar, 0.0), out[4]);
    try std.testing.expectEqual(@as(Scalar, 0.0), out[5]);
    try std.testing.expectEqual(@as(Scalar, 1.0), out[6]);
    try std.testing.expectEqual(@as(Scalar, 4.0), out[9]);
}

test "sum reduction" {
    const allocator = std.testing.allocator;
    const n = 100;

    const x = try allocator.alloc(Scalar, n);
    defer allocator.free(x);

    for (x, 0..) |*val, i| val.* = @floatFromInt(i);

    const result = sum(x);
    const expected = @as(Scalar, @floatFromInt(n * (n - 1) / 2));
    try std.testing.expectApproxEqAbs(expected, result, 1e-3);
}

test "max reduction" {
    const allocator = std.testing.allocator;
    const n = 100;

    const x = try allocator.alloc(Scalar, n);
    defer allocator.free(x);

    for (x, 0..) |*val, i| val.* = @floatFromInt(i);

    const result = max(x);
    try std.testing.expectEqual(@as(Scalar, 99.0), result);
}

/// Broadcast add: out = a + b with broadcasting support
/// Supports shapes like [batch, features] + [features] or [batch, features] + [1, features]
pub fn broadcastAddInto(
    out: []Scalar,
    a: []const Scalar,
    b: []const Scalar,
    a_shape: []const usize,
    b_shape: []const usize,
) !void {
    // Simple case: same shapes (no broadcasting needed)
    if (a.len == b.len and a_shape.len == b_shape.len) {
        var same_shape = true;
        for (a_shape, 0..) |dim, i| {
            if (dim != b_shape[i]) {
                same_shape = false;
                break;
            }
        }
        if (same_shape) {
            return addInto(out, a, b);
        }
    }

    // Handle [batch, features] + [features] case
    if (a_shape.len == 2 and b_shape.len == 1) {
        const batch = a_shape[0];
        const features = a_shape[1];

        if (b_shape[0] != features) {
            return error.ShapeMismatch;
        }

        // For each batch item, add the bias vector
        for (0..batch) |i| {
            const offset = i * features;
            const vec_count = features / simd_lanes;
            const remainder = features % simd_lanes;

            // SIMD path
            for (0..vec_count) |v| {
                const idx = offset + v * simd_lanes;
                const a_vec: Vec = a[idx..][0..simd_lanes].*;
                const b_vec: Vec = b[v * simd_lanes ..][0..simd_lanes].*;
                const result_vec = a_vec + b_vec;
                const result_array: [simd_lanes]Scalar = result_vec;
                @memcpy(out[idx..][0..simd_lanes], &result_array);
            }

            // Scalar tail
            const tail_offset = vec_count * simd_lanes;
            for (0..remainder) |j| {
                out[offset + tail_offset + j] = a[offset + tail_offset + j] + b[tail_offset + j];
            }
        }

        return;
    }

    // Handle [batch, features] + [1, features] case
    if (a_shape.len == 2 and b_shape.len == 2 and b_shape[0] == 1) {
        const batch = a_shape[0];
        const features = a_shape[1];

        if (b_shape[1] != features) {
            return error.ShapeMismatch;
        }

        // Same as above, but b is laid out as [1, features]
        for (0..batch) |i| {
            const offset = i * features;
            const vec_count = features / simd_lanes;
            const remainder = features % simd_lanes;

            // SIMD path
            for (0..vec_count) |v| {
                const idx = offset + v * simd_lanes;
                const a_vec: Vec = a[idx..][0..simd_lanes].*;
                const b_vec: Vec = b[v * simd_lanes ..][0..simd_lanes].*;
                const result_vec = a_vec + b_vec;
                const result_array: [simd_lanes]Scalar = result_vec;
                @memcpy(out[idx..][0..simd_lanes], &result_array);
            }

            // Scalar tail
            const tail_offset = vec_count * simd_lanes;
            for (0..remainder) |j| {
                out[offset + tail_offset + j] = a[offset + tail_offset + j] + b[tail_offset + j];
            }
        }

        return;
    }

    return error.UnsupportedBroadcast;
}

/// Reduce mean over all elements
pub fn mean(x: []const Scalar) Scalar {
    if (x.len == 0) return 0.0;
    const total = sum(x);
    return total / @as(Scalar, @floatFromInt(x.len));
}

test "broadcast add 2d + 1d" {
    const allocator = std.testing.allocator;

    // [2, 3] + [3]
    const a = try allocator.alloc(Scalar, 6);
    defer allocator.free(a);
    const b = try allocator.alloc(Scalar, 3);
    defer allocator.free(b);
    const out = try allocator.alloc(Scalar, 6);
    defer allocator.free(out);

    // a = [[1, 2, 3],
    //      [4, 5, 6]]
    for (a, 0..) |*val, i| val.* = @floatFromInt(i + 1);

    // b = [10, 20, 30]
    b[0] = 10.0;
    b[1] = 20.0;
    b[2] = 30.0;

    const a_shape = [_]usize{ 2, 3 };
    const b_shape = [_]usize{3};

    try broadcastAddInto(out, a, b, &a_shape, &b_shape);

    // Expected: [[11, 22, 33], [14, 25, 36]]
    try std.testing.expectApproxEqAbs(@as(Scalar, 11.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 22.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 33.0), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 14.0), out[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 25.0), out[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 36.0), out[5], 1e-5);
}

test "mean reduction" {
    const allocator = std.testing.allocator;

    const x = try allocator.alloc(Scalar, 5);
    defer allocator.free(x);

    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;
    x[4] = 5.0;

    const result = mean(x);
    try std.testing.expectApproxEqAbs(@as(Scalar, 3.0), result, 1e-5);
}

/// Gather specific actions from Q-values tensor for DQN
/// q_values: [batch_size, num_actions] - flattened row-major
/// actions: [batch_size] - action indices
/// out: [batch_size] - selected Q-values
pub fn gatherActions(
    out: []Scalar,
    q_values: []const Scalar,
    actions: []const u8,
    batch_size: usize,
    num_actions: usize,
) !void {
    if (q_values.len != batch_size * num_actions) {
        return error.ShapeMismatch;
    }
    if (out.len != batch_size or actions.len != batch_size) {
        return error.ShapeMismatch;
    }

    for (0..batch_size) |i| {
        const action = actions[i];
        if (action >= num_actions) {
            return error.InvalidAction;
        }
        const offset = i * num_actions + action;
        out[i] = q_values[offset];
    }
}

/// Max reduction along axis 1 for 2D tensor
/// input: [batch_size, features] - flattened row-major
/// out: [batch_size] - max value along axis 1
pub fn maxAlongAxis1(
    out: []Scalar,
    input: []const Scalar,
    batch_size: usize,
    features: usize,
) !void {
    if (input.len != batch_size * features) {
        return error.ShapeMismatch;
    }
    if (out.len != batch_size) {
        return error.ShapeMismatch;
    }

    for (0..batch_size) |i| {
        const row_start = i * features;
        const row = input[row_start .. row_start + features];
        out[i] = max(row);
    }
}

test "gather actions" {
    const allocator = std.testing.allocator;

    // Q-values: [3 batch × 4 actions]
    const q_values = try allocator.alloc(Scalar, 12);
    defer allocator.free(q_values);

    // Fill with test values
    for (q_values, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    const actions = [_]u8{ 1, 2, 0 };
    var out: [3]Scalar = undefined;

    try gatherActions(&out, q_values, &actions, 3, 4);

    // q_values[0, 1] = 1, q_values[1, 2] = 6, q_values[2, 0] = 8
    try std.testing.expectEqual(@as(Scalar, 1.0), out[0]);
    try std.testing.expectEqual(@as(Scalar, 6.0), out[1]);
    try std.testing.expectEqual(@as(Scalar, 8.0), out[2]);
}

test "max along axis 1" {
    // Input: [2 × 3]
    const input = [_]Scalar{ 1.0, 5.0, 3.0, 2.0, 1.0, 4.0 };
    var out: [2]Scalar = undefined;

    try maxAlongAxis1(&out, &input, 2, 3);

    try std.testing.expectEqual(@as(Scalar, 5.0), out[0]); // max of [1, 5, 3]
    try std.testing.expectEqual(@as(Scalar, 4.0), out[1]); // max of [2, 1, 4]
}
