const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const nn_mod = @import("../nn.zig");

const TensorContext = tensor_mod.TensorContext;
const GradContext = tensor_mod.GradContext;
const AutodiffContext = tensor_mod.AutodiffContext;
const TrackedTensor = tensor_mod.TrackedTensor;
const Model = nn_mod.Model;

/// Q-network wrapper managing online and target networks
pub const QNetwork = struct {
    online_model: Model,
    target_model: Model,
    allocator: std.mem.Allocator,

    /// Initialize Q-networks with the same architecture
    /// Architecture: [324 (state) → hidden layers → 12 (Q-values)]
    pub fn init(
        allocator: std.mem.Allocator,
        layer_sizes: []const usize,
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        rng: std.Random,
    ) !QNetwork {
        const online_model = try Model.init(
            allocator,
            layer_sizes,
            tensor_ctx,
            grad_ctx,
            rng,
        );

        const target_model = try Model.init(
            allocator,
            layer_sizes,
            tensor_ctx,
            grad_ctx,
            rng,
        );

        return QNetwork{
            .online_model = online_model,
            .target_model = target_model,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *QNetwork) void {
        self.online_model.deinit();
        self.target_model.deinit();
    }

    /// Predict Q-values for a batch of states
    /// state_batch: [batch_size, 324]
    /// Returns: TrackedTensor [batch_size, 12]
    pub fn predict(
        model: *Model,
        state_batch: TrackedTensor,
        ad_ctx: *AutodiffContext,
        tensor_ctx: *TensorContext,
    ) !TrackedTensor {
        return model.forward(state_batch, ad_ctx, tensor_ctx);
    }

    /// Hard copy online weights to target network (tau = 1.0)
    pub fn updateTargetHard(self: *QNetwork) void {
        // Copy weights layer by layer
        for (self.online_model.layers, 0..) |*online_layer, i| {
            const target_layer = &self.target_model.layers[i];

            // Copy W
            @memcpy(target_layer.W.data, online_layer.W.data);

            // Copy b
            @memcpy(target_layer.b.data, online_layer.b.data);
        }
    }

    /// Polyak averaging update: target = tau * online + (1 - tau) * target
    pub fn updateTargetPolyak(self: *QNetwork, tau: f32) void {
        const one_minus_tau = 1.0 - tau;

        for (self.online_model.layers, 0..) |*online_layer, i| {
            const target_layer = &self.target_model.layers[i];

            // Update W
            for (online_layer.W.data, 0..) |online_val, j| {
                target_layer.W.data[j] = tau * online_val + one_minus_tau * target_layer.W.data[j];
            }

            // Update b
            for (online_layer.b.data, 0..) |online_val, j| {
                target_layer.b.data[j] = tau * online_val + one_minus_tau * target_layer.b.data[j];
            }
        }
    }
};

test "qnetwork init and update target" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Small network: 324 → 64 → 12
    const layer_sizes = [_]usize{ 324, 64, 12 };
    var qnet = try QNetwork.init(
        std.testing.allocator,
        &layer_sizes,
        &tensor_ctx,
        &grad_ctx,
        rng,
    );
    defer qnet.deinit();

    // Check initial weights are different (random init)
    const online_w0 = qnet.online_model.layers[0].W.data[0];
    const target_w0 = qnet.target_model.layers[0].W.data[0];
    try std.testing.expect(online_w0 != target_w0);

    // Hard update
    qnet.updateTargetHard();

    // Now they should match
    const updated_target_w0 = qnet.target_model.layers[0].W.data[0];
    try std.testing.expectEqual(online_w0, updated_target_w0);
}
