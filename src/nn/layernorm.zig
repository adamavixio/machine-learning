const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const TensorContext = tensor_mod.TensorContext;
const GradContext = tensor_mod.GradContext;
const AutodiffContext = tensor_mod.AutodiffContext;
const TrackedTensor = tensor_mod.TrackedTensor;
const GradHandle = tensor_mod.GradHandle;
const Scalar = tensor_mod.TensorConfig.Scalar;

/// Layer Normalization layer
///
/// Normalizes activations across features: output = gamma * (x - mean) / sqrt(var + eps) + beta
/// where gamma and beta are learnable parameters
pub const LayerNorm = struct {
    /// Scale parameter (gamma): [normalized_shape]
    gamma: Tensor,
    gamma_handle: GradHandle,

    /// Shift parameter (beta): [normalized_shape]
    beta: Tensor,
    beta_handle: GradHandle,

    normalized_shape: usize,
    eps: Scalar,

    /// Initialize LayerNorm layer
    /// normalized_shape: Number of features to normalize (e.g., hidden_dim)
    /// eps: Small constant for numerical stability (default 1e-5)
    pub fn init(
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        normalized_shape: usize,
        eps: Scalar,
    ) !LayerNorm {
        // Allocate gamma (initialized to 1.0)
        const gamma = try tensor_ctx.allocTensor(&[_]usize{normalized_shape});
        for (gamma.data) |*val| {
            val.* = 1.0;
        }

        // Allocate beta (initialized to 0.0)
        const beta = try tensor_ctx.allocTensor(&[_]usize{normalized_shape});
        @memset(beta.data, 0.0);

        // Register gradients
        const gamma_handle = try grad_ctx.allocGrad(gamma.shape);
        const beta_handle = try grad_ctx.allocGrad(beta.shape);

        return LayerNorm{
            .gamma = gamma,
            .gamma_handle = gamma_handle,
            .beta = beta,
            .beta_handle = beta_handle,
            .normalized_shape = normalized_shape,
            .eps = eps,
        };
    }

    /// Forward pass: normalize input
    /// x: [batch_size, normalized_shape]
    /// Returns: [batch_size, normalized_shape]
    pub fn forward(
        self: *LayerNorm,
        x: TrackedTensor,
        ad_ctx: *AutodiffContext,
        tensor_ctx: *TensorContext,
    ) !TrackedTensor {
        const batch_size = if (x.shape().dims.len == 2) x.shape().dims[0] else 1;
        const feature_dim = if (x.shape().dims.len == 2) x.shape().dims[1] else x.shape().dims[0];

        if (feature_dim != self.normalized_shape) {
            return error.InvalidDimensions;
        }

        // Allocate output
        const output = try tensor_ctx.allocTensor(&[_]usize{ batch_size, feature_dim });

        // Get input data
        const x_data = x.data();

        // Process each sample in batch
        var batch_idx: usize = 0;
        while (batch_idx < batch_size) : (batch_idx += 1) {
            const start_idx = batch_idx * feature_dim;
            const end_idx = start_idx + feature_dim;
            const x_row = x_data[start_idx..end_idx];

            // Compute mean
            var mean: Scalar = 0.0;
            for (x_row) |val| {
                mean += val;
            }
            mean /= @as(Scalar, @floatFromInt(feature_dim));

            // Compute variance
            var variance: Scalar = 0.0;
            for (x_row) |val| {
                const diff = val - mean;
                variance += diff * diff;
            }
            variance /= @as(Scalar, @floatFromInt(feature_dim));

            // Normalize and apply affine transform
            const std_dev = @sqrt(variance + self.eps);
            for (0..feature_dim) |i| {
                const normalized = (x_row[i] - mean) / std_dev;
                output.data[start_idx + i] = self.gamma.data[i] * normalized + self.beta.data[i];
            }
        }

        // TODO: Properly track this operation in autodiff context
        // For now, return untracked tensor (will need backward pass implementation)
        return ad_ctx.track(output);
    }

    /// Backward pass
    ///
    /// Gradient computation for LayerNorm:
    /// d_x = (1/N*std) * (N * d_out * gamma - sum(d_out * gamma) - normalized * sum(d_out * gamma * normalized))
    /// d_gamma = sum(d_out * normalized)
    /// d_beta = sum(d_out)
    ///
    /// where:
    /// - N = normalized_shape
    /// - std = sqrt(var + eps)
    /// - normalized = (x - mean) / std
    pub fn backward(
        self: *LayerNorm,
        grad_output: []const Scalar,
        x: []const Scalar,
        grad_input: []Scalar,
        grad_ctx: *GradContext,
    ) !void {
        const batch_size = grad_output.len / self.normalized_shape;
        const feature_dim = self.normalized_shape;

        // Get gradient buffers
        var grad_gamma = try grad_ctx.getGrad(self.gamma_handle);
        var grad_beta = try grad_ctx.getGrad(self.beta_handle);

        // Initialize parameter gradients to zero
        @memset(grad_gamma, 0.0);
        @memset(grad_beta, 0.0);

        // Process each sample in batch
        var batch_idx: usize = 0;
        while (batch_idx < batch_size) : (batch_idx += 1) {
            const start_idx = batch_idx * feature_dim;
            const end_idx = start_idx + feature_dim;
            const x_row = x[start_idx..end_idx];
            const grad_out_row = grad_output[start_idx..end_idx];

            // Recompute mean and variance (same as forward pass)
            var mean: Scalar = 0.0;
            for (x_row) |val| {
                mean += val;
            }
            mean /= @as(Scalar, @floatFromInt(feature_dim));

            var variance: Scalar = 0.0;
            for (x_row) |val| {
                const diff = val - mean;
                variance += diff * diff;
            }
            variance /= @as(Scalar, @floatFromInt(feature_dim));

            const std_dev = @sqrt(variance + self.eps);

            // Compute intermediate values
            var sum_grad_out_gamma: Scalar = 0.0;
            var sum_grad_out_gamma_normalized: Scalar = 0.0;

            for (0..feature_dim) |i| {
                const normalized = (x_row[i] - mean) / std_dev;
                sum_grad_out_gamma += grad_out_row[i] * self.gamma.data[i];
                sum_grad_out_gamma_normalized += grad_out_row[i] * self.gamma.data[i] * normalized;

                // Accumulate gradients for gamma and beta
                grad_gamma[i] += grad_out_row[i] * normalized;
                grad_beta[i] += grad_out_row[i];
            }

            // Compute gradient w.r.t. input
            const N_f = @as(Scalar, @floatFromInt(feature_dim));
            const inv_N_std = 1.0 / (N_f * std_dev);

            for (0..feature_dim) |i| {
                const normalized = (x_row[i] - mean) / std_dev;
                grad_input[start_idx + i] = inv_N_std * (
                    N_f * grad_out_row[i] * self.gamma.data[i] -
                    sum_grad_out_gamma -
                    normalized * sum_grad_out_gamma_normalized
                );
            }
        }
    }
};

test "layernorm init" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    const layer_norm = try LayerNorm.init(&tensor_ctx, &grad_ctx, 128, 1e-5);

    // Check gamma initialized to 1.0
    try std.testing.expectEqual(@as(Scalar, 1.0), layer_norm.gamma.data[0]);
    try std.testing.expectEqual(@as(Scalar, 1.0), layer_norm.gamma.data[127]);

    // Check beta initialized to 0.0
    try std.testing.expectEqual(@as(Scalar, 0.0), layer_norm.beta.data[0]);
    try std.testing.expectEqual(@as(Scalar, 0.0), layer_norm.beta.data[127]);
}

test "layernorm forward basic" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var ad_ctx = AutodiffContext.init(std.testing.allocator, &grad_ctx);
    defer ad_ctx.deinit();

    var layer_norm = try LayerNorm.init(&tensor_ctx, &grad_ctx, 4, 1e-5);

    // Create simple input [1, 4]: [1.0, 2.0, 3.0, 4.0]
    const x = try tensor_ctx.allocTensor(&[_]usize{ 1, 4 });
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;

    const x_tracked = ad_ctx.track(x);

    // Forward pass
    const output = try layer_norm.forward(x_tracked, &ad_ctx, &tensor_ctx);

    // Mean should be 2.5
    // Variance should be 1.25
    // Output should be normalized (approximately [-1.34, -0.45, 0.45, 1.34] with gamma=1, beta=0)

    // Check output shape
    try std.testing.expectEqual(@as(usize, 1), output.shape().dims[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape().dims[1]);

    // Check approximate values (normalized distribution should have mean ~0, std ~1)
    const out_data = output.data();
    const mean = (out_data[0] + out_data[1] + out_data[2] + out_data[3]) / 4.0;
    try std.testing.expect(@abs(mean) < 0.01); // Mean should be near 0
}

test "layernorm backward" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var layer_norm = try LayerNorm.init(&tensor_ctx, &grad_ctx, 4, 1e-5);

    // Create input [1, 4]: [1.0, 2.0, 3.0, 4.0]
    const x = try tensor_ctx.allocTensor(&[_]usize{ 1, 4 });
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;

    // Create gradient output (all ones)
    const grad_out = try tensor_ctx.allocTensor(&[_]usize{ 1, 4 });
    @memset(grad_out.data, 1.0);

    // Create gradient input buffer
    const grad_in = try tensor_ctx.allocTensor(&[_]usize{ 1, 4 });

    // Run backward pass
    try layer_norm.backward(grad_out.data, x.data, grad_in.data, &grad_ctx);

    // Check that grad_beta is computed (should be all 1.0 since grad_out is all 1.0)
    const grad_beta = try grad_ctx.getGrad(layer_norm.beta_handle);
    for (grad_beta) |gb| {
        try std.testing.expectApproxEqRel(@as(Scalar, 1.0), gb, 0.01);
    }

    // Check that grad_gamma is computed (should be equal to normalized values)
    const grad_gamma = try grad_ctx.getGrad(layer_norm.gamma_handle);
    // With input [1,2,3,4], mean=2.5, std≈1.118
    // normalized ≈ [-1.34, -0.45, 0.45, 1.34]
    // grad_gamma should be approximately these values
    try std.testing.expect(@abs(grad_gamma[0] + grad_gamma[3]) < 0.01); // Should be symmetric
    try std.testing.expect(@abs(grad_gamma[1] + grad_gamma[2]) < 0.01); // Should be symmetric

    // Check that grad_input sums to approximately zero (property of LayerNorm backward)
    var sum: Scalar = 0.0;
    for (grad_in.data) |gi| {
        sum += gi;
    }
    try std.testing.expect(@abs(sum) < 0.01);
}
