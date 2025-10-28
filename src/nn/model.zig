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
/// Supports optional dueling architecture with separate value and advantage heads
pub const Model = struct {
    layers: []DenseLayer,
    param_handles: []tensor_mod.GradHandle,
    allocator: std.mem.Allocator,
    // Dueling architecture support
    use_dueling: bool,
    value_head: ?DenseLayer,
    advantage_head: ?DenseLayer,
    num_actions: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        layer_sizes: []const usize,
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        rng: std.Random,
    ) !Model {
        return initWithDueling(allocator, layer_sizes, tensor_ctx, grad_ctx, rng, false);
    }

    pub fn initWithDueling(
        allocator: std.mem.Allocator,
        layer_sizes: []const usize,
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        rng: std.Random,
        use_dueling: bool,
    ) !Model {
        if (layer_sizes.len < 2) {
            return error.InvalidLayerSizes;
        }

        if (use_dueling) {
            // Dueling architecture: layer_sizes represents shared backbone
            // Last element is num_actions, we'll create separate value/advantage heads
            const num_actions = layer_sizes[layer_sizes.len - 1];
            const num_layers = layer_sizes.len - 2; // Exclude input and output
            const layers = try allocator.alloc(DenseLayer, num_layers);

            // Build shared backbone (all layers except the last output layer)
            for (0..num_layers) |i| {
                layers[i] = try DenseLayer.init(
                    allocator,
                    tensor_ctx,
                    grad_ctx,
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    rng,
                );
            }

            // Create separate heads from last hidden layer
            const last_hidden_size = layer_sizes[layer_sizes.len - 2];

            // Value head: last_hidden → 1
            const value_head = try DenseLayer.init(
                allocator,
                tensor_ctx,
                grad_ctx,
                last_hidden_size,
                1,
                rng,
            );

            // Advantage head: last_hidden → num_actions
            const advantage_head = try DenseLayer.init(
                allocator,
                tensor_ctx,
                grad_ctx,
                last_hidden_size,
                num_actions,
                rng,
            );

            // Build flat list of parameter handles: backbone + value head + advantage head
            const total_params = (num_layers + 2) * 2; // Each layer has W and b
            const param_handles = try allocator.alloc(tensor_mod.GradHandle, total_params);

            // Backbone parameters
            for (layers, 0..) |layer, i| {
                param_handles[i * 2] = layer.W_handle;
                param_handles[i * 2 + 1] = layer.b_handle;
            }

            // Value head parameters
            const value_offset = num_layers * 2;
            param_handles[value_offset] = value_head.W_handle;
            param_handles[value_offset + 1] = value_head.b_handle;

            // Advantage head parameters
            const adv_offset = value_offset + 2;
            param_handles[adv_offset] = advantage_head.W_handle;
            param_handles[adv_offset + 1] = advantage_head.b_handle;

            return Model{
                .layers = layers,
                .param_handles = param_handles,
                .allocator = allocator,
                .use_dueling = true,
                .value_head = value_head,
                .advantage_head = advantage_head,
                .num_actions = num_actions,
            };
        } else {
            // Standard architecture: full sequential layers
            const num_layers = layer_sizes.len - 1;
            const layers = try allocator.alloc(DenseLayer, num_layers);

            for (0..num_layers) |i| {
                layers[i] = try DenseLayer.init(
                    allocator,
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

            const num_actions = layer_sizes[layer_sizes.len - 1];

            return Model{
                .layers = layers,
                .param_handles = param_handles,
                .allocator = allocator,
                .use_dueling = false,
                .value_head = null,
                .advantage_head = null,
                .num_actions = num_actions,
            };
        }
    }

    pub fn deinit(self: *Model) void {
        // Clean up scratch buffers for each layer
        for (self.layers) |*layer| {
            layer.deinit();
        }
        // Clean up dueling heads if present
        if (self.value_head) |*vh| {
            vh.deinit();
        }
        if (self.advantage_head) |*ah| {
            ah.deinit();
        }
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

        // Backbone parameters
        for (self.layers, 0..) |layer, i| {
            params[i * 2] = layer.W;
            params[i * 2 + 1] = layer.b;
        }

        // Dueling heads parameters if present
        if (self.use_dueling) {
            const value_offset = self.layers.len * 2;
            if (self.value_head) |vh| {
                params[value_offset] = vh.W;
                params[value_offset + 1] = vh.b;
            }
            if (self.advantage_head) |ah| {
                const adv_offset = value_offset + 2;
                params[adv_offset] = ah.W;
                params[adv_offset + 1] = ah.b;
            }
        }

        return params;
    }

    /// Forward pass through all layers
    /// For dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    pub fn forward(
        self: *Model,
        x: TrackedTensor,
        ad_ctx: *AutodiffContext,
        tensor_ctx: *TensorContext,
    ) !TrackedTensor {
        var current = x;

        // Pass through shared backbone
        for (self.layers) |*layer| {
            current = try layer.forward(current, ad_ctx, tensor_ctx);
        }

        if (self.use_dueling) {
            // Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
            const batch_size = current.shape().dims[0];

            // Value head: [batch_size, hidden] → [batch_size, 1]
            const value = try self.value_head.?.forward(current, ad_ctx, tensor_ctx);

            // Advantage head: [batch_size, hidden] → [batch_size, num_actions]
            const advantage = try self.advantage_head.?.forward(current, ad_ctx, tensor_ctx);

            // Manually combine V and A: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            // This mean-centering ensures Q-values are well-behaved
            const q_values_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.num_actions });

            const v_data = value.data();
            const adv_data = advantage.data();
            const q_data = q_values_tensor.data;

            // Forward pass: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            for (0..batch_size) |i| {
                const v = v_data[i];

                // Compute mean advantage for this sample
                var mean_adv: f32 = 0.0;
                for (0..self.num_actions) |a| {
                    mean_adv += adv_data[i * self.num_actions + a];
                }
                mean_adv /= @as(f32, @floatFromInt(self.num_actions));

                // Q(s,a) = V(s) + (A(s,a) - mean(A))
                for (0..self.num_actions) |a| {
                    q_data[i * self.num_actions + a] = v + (adv_data[i * self.num_actions + a] - mean_adv);
                }
            }

            // Register operation with autodiff system
            // This connects the output Q-values to the input value and advantage
            const q_values = try ad_ctx.track(q_values_tensor);
            const operands = [_]tensor_mod.GradHandle{ value.grad_handle, advantage.grad_handle };
            try ad_ctx.recordOp(.dueling_q, q_values.grad_handle, &operands, .{
                .dueling_q = .{
                    .batch_size = batch_size,
                    .num_actions = self.num_actions,
                },
            });

            return q_values;
        } else {
            // Standard sequential forward pass (already computed in backbone loop)
            return current;
        }
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
