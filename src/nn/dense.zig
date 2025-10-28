const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const TensorContext = tensor_mod.TensorContext;
const GradContext = tensor_mod.GradContext;
const AutodiffContext = tensor_mod.AutodiffContext;
const TrackedTensor = tensor_mod.TrackedTensor;
const GradHandle = tensor_mod.GradHandle;
const Scalar = tensor_mod.TensorConfig.Scalar;

/// Dense (fully connected) layer with weight matrix and bias vector
pub const DenseLayer = struct {
    /// Weight matrix: [in_dim x out_dim]
    W: Tensor,
    W_handle: GradHandle,

    /// Bias vector: [out_dim]
    b: Tensor,
    b_handle: GradHandle,

    in_dim: usize,
    out_dim: usize,

    /// Scratch buffer for backward pass gradient computation
    /// Pre-allocated to avoid allocator overhead during training
    /// Size: [max_batch_size x in_dim] for dA (gradient w.r.t. input)
    scratch_dA: []Scalar,

    /// Scratch buffer for backward pass gradient computation
    /// Pre-allocated to avoid allocator overhead during training
    /// Size: [in_dim x out_dim] for dB (gradient w.r.t. weights)
    scratch_dB: []Scalar,

    /// Allocator used for scratch buffers (needed for cleanup)
    allocator: std.mem.Allocator,

    /// Maximum batch size for scratch buffer allocation
    const MAX_BATCH_SIZE: usize = 64;

    /// Initialize a dense layer with He initialization
    pub fn init(
        allocator: std.mem.Allocator,
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        in_dim: usize,
        out_dim: usize,
        rng: std.Random,
    ) !DenseLayer {
        // Allocate weight matrix [in_dim x out_dim]
        const W = try tensor_ctx.allocTensor(&[_]usize{ in_dim, out_dim });

        // Allocate bias vector [out_dim]
        const b = try tensor_ctx.allocTensor(&[_]usize{out_dim});

        // He initialization: Normal(0, sqrt(2 / in_dim))
        const he_std = @sqrt(2.0 / @as(Scalar, @floatFromInt(in_dim)));

        // Initialize weights
        for (W.data) |*val| {
            const uniform1 = rng.float(Scalar);
            const uniform2 = rng.float(Scalar);
            // Box-Muller transform for normal distribution
            const z = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * std.math.pi * uniform2);
            val.* = z * he_std;
        }

        // Initialize bias to zero
        @memset(b.data, 0.0);

        // Register gradients
        const W_handle = try grad_ctx.allocGrad(W.shape);
        const b_handle = try grad_ctx.allocGrad(b.shape);

        // Allocate scratch buffers for backward pass
        // dA: [max_batch_size x in_dim] for gradient w.r.t. input
        const scratch_dA = try allocator.alloc(Scalar, MAX_BATCH_SIZE * in_dim);
        // dB: [in_dim x out_dim] for gradient w.r.t. weights
        const scratch_dB = try allocator.alloc(Scalar, in_dim * out_dim);

        return DenseLayer{
            .W = W,
            .W_handle = W_handle,
            .b = b,
            .b_handle = b_handle,
            .in_dim = in_dim,
            .out_dim = out_dim,
            .scratch_dA = scratch_dA,
            .scratch_dB = scratch_dB,
            .allocator = allocator,
        };
    }

    /// Clean up scratch buffers
    pub fn deinit(self: *DenseLayer) void {
        self.allocator.free(self.scratch_dA);
        self.allocator.free(self.scratch_dB);
    }

    /// Forward pass: y = x @ W + b
    pub fn forward(
        self: *DenseLayer,
        x: TrackedTensor,
        ad_ctx: *AutodiffContext,
        tensor_ctx: *TensorContext,
    ) !TrackedTensor {
        // Get batch size from input
        const batch_size = if (x.shape().dims.len == 2) x.shape().dims[0] else 1;

        // Validate dimensions
        const x_cols = if (x.shape().dims.len == 2) x.shape().dims[1] else x.shape().dims[0];
        if (x_cols != self.in_dim) {
            return error.ShapeMismatch;
        }

        // Track weights and bias
        const W_tracked = TrackedTensor{
            .tensor = self.W,
            .grad_handle = self.W_handle,
        };

        const b_tracked = TrackedTensor{
            .tensor = self.b,
            .grad_handle = self.b_handle,
        };

        // Compute matmul: xW = x @ W
        const xW_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.out_dim });

        // Slice scratch buffers to the correct size for this batch
        // scratch_dA: [batch_size x in_dim] for gradient w.r.t. input (x)
        const scratch_dA_slice = self.scratch_dA[0 .. batch_size * self.in_dim];
        // scratch_dB: [in_dim x out_dim] for gradient w.r.t. weights (W)
        const scratch_dB_slice = self.scratch_dB[0 .. self.in_dim * self.out_dim];

        const xW = try ad_ctx.trackedMatmul(
            x,
            W_tracked,
            batch_size,
            self.in_dim,
            self.out_dim,
            xW_tensor,
            scratch_dA_slice,
            scratch_dB_slice,
        );

        // Add bias: y = xW + b (broadcast b across batch)
        const y_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.out_dim });
        const y = try ad_ctx.trackedBroadcastAdd(xW, b_tracked, y_tensor);

        return y;
    }
};

test "dense layer init" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var layer = try DenseLayer.init(std.testing.allocator, &tensor_ctx, &grad_ctx, 10, 5, rng);
    defer layer.deinit();

    try std.testing.expectEqual(@as(usize, 10), layer.in_dim);
    try std.testing.expectEqual(@as(usize, 5), layer.out_dim);
    try std.testing.expectEqual(@as(usize, 50), layer.W.data.len);
    try std.testing.expectEqual(@as(usize, 5), layer.b.data.len);

    // Check bias is zero-initialized
    for (layer.b.data) |val| {
        try std.testing.expectEqual(@as(Scalar, 0.0), val);
    }

    // Check weights are non-zero (initialized)
    var non_zero_count: usize = 0;
    for (layer.W.data) |val| {
        if (val != 0.0) non_zero_count += 1;
    }
    try std.testing.expect(non_zero_count > 0);
}
