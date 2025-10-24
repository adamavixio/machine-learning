const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const dense_mod = @import("dense.zig");

const Tensor = tensor_mod.Tensor;
const TensorContext = tensor_mod.TensorContext;
const GradContext = tensor_mod.GradContext;
const AutodiffContext = tensor_mod.AutodiffContext;
const TrackedTensor = tensor_mod.TrackedTensor;
const DenseLayer = dense_mod.DenseLayer;

/// Sequential model composed of multiple dense layers
pub const Model = struct {
    layers: []DenseLayer,
    param_handles: []tensor_mod.GradHandle,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        layer_sizes: []const usize,
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        rng: std.Random,
    ) !Model {
        if (layer_sizes.len < 2) {
            return error.InvalidLayerSizes;
        }

        const num_layers = layer_sizes.len - 1;
        const layers = try allocator.alloc(DenseLayer, num_layers);

        for (0..num_layers) |i| {
            layers[i] = try DenseLayer.init(
                tensor_ctx,
                grad_ctx,
                layer_sizes[i],
                layer_sizes[i + 1],
                rng,
            );
        }

        // Build flat list of parameter handles (2 per layer: W and b)
        const param_handles = try allocator.alloc(tensor_mod.GradHandle, num_layers * 2);
        for (layers, 0..) |layer, i| {
            param_handles[i * 2] = layer.W_handle;
            param_handles[i * 2 + 1] = layer.b_handle;
        }

        return Model{
            .layers = layers,
            .param_handles = param_handles,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.layers);
        self.allocator.free(self.param_handles);
    }

    /// Get all parameter handles for optimizer
    pub fn getParameterHandles(self: *Model) []tensor_mod.GradHandle {
        return self.param_handles;
    }

    /// Get all parameter tensors (for in-place updates)
    /// Returns: slice of Tensors (W, b, W, b, ...)
    /// Caller must free the returned slice
    pub fn getParameterTensors(self: *Model, allocator: std.mem.Allocator) ![]Tensor {
        const params = try allocator.alloc(Tensor, self.param_handles.len);
        for (self.layers, 0..) |layer, i| {
            params[i * 2] = layer.W;
            params[i * 2 + 1] = layer.b;
        }
        return params;
    }

    /// Forward pass through all layers
    /// Note: This is a simplified version that doesn't handle activations yet
    pub fn forward(
        self: *Model,
        x: TrackedTensor,
        ad_ctx: *AutodiffContext,
        tensor_ctx: *TensorContext,
    ) !TrackedTensor {
        var current = x;

        for (self.layers) |*layer| {
            current = try layer.forward(current, ad_ctx, tensor_ctx);
        }

        return current;
    }
};

test "model init" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const layer_sizes = [_]usize{ 10, 20, 5 };
    var model = try Model.init(
        std.testing.allocator,
        &layer_sizes,
        &tensor_ctx,
        &grad_ctx,
        rng,
    );
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 2), model.layers.len);
    try std.testing.expectEqual(@as(usize, 10), model.layers[0].in_dim);
    try std.testing.expectEqual(@as(usize, 20), model.layers[0].out_dim);
    try std.testing.expectEqual(@as(usize, 20), model.layers[1].in_dim);
    try std.testing.expectEqual(@as(usize, 5), model.layers[1].out_dim);
}
