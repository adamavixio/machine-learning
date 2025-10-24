const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const nn_mod = @import("../nn.zig");
const env_mod = @import("../env.zig");
const replay_mod = @import("replay.zig");
const qnetwork_mod = @import("qnetwork.zig");

const TensorContext = tensor_mod.TensorContext;
const GradContext = tensor_mod.GradContext;
const AutodiffContext = tensor_mod.AutodiffContext;
const TrackedTensor = tensor_mod.TrackedTensor;
const Tensor = tensor_mod.Tensor;
const CubeState = env_mod.CubeState;
const Move = env_mod.Move;
const ReplayBuffer = replay_mod.ReplayBuffer;
const Experience = replay_mod.Experience;
const QNetwork = qnetwork_mod.QNetwork;

/// DQN hyperparameters
pub const DQNConfig = struct {
    gamma: f32 = 0.99, // Discount factor
    epsilon_start: f32 = 1.0,
    epsilon_end: f32 = 0.01,
    epsilon_decay: f32 = 0.995,
    learning_rate: f32 = 0.001,
    batch_size: usize = 32,
    replay_buffer_size: usize = 10000,
    target_update_freq: usize = 100, // Update target every N steps
    max_episode_steps: usize = 100,
    scramble_depth: usize = 5, // Start with easy scrambles
};

/// DQN agent
pub const DQNAgent = struct {
    qnet: QNetwork,
    replay_buffer: ReplayBuffer,
    config: DQNConfig,
    epsilon: f32,
    step_count: usize,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        layer_sizes: []const usize,
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        config: DQNConfig,
        rng: std.Random,
    ) !DQNAgent {
        const qnet = try QNetwork.init(
            allocator,
            layer_sizes,
            tensor_ctx,
            grad_ctx,
            rng,
        );

        const replay_buffer = try ReplayBuffer.init(allocator, config.replay_buffer_size);

        return DQNAgent{
            .qnet = qnet,
            .replay_buffer = replay_buffer,
            .config = config,
            .epsilon = config.epsilon_start,
            .step_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DQNAgent) void {
        self.qnet.deinit();
        self.replay_buffer.deinit();
    }

    /// Epsilon-greedy action selection
    pub fn selectAction(
        self: *DQNAgent,
        state: CubeState,
        tensor_ctx: *TensorContext,
        ad_ctx: *AutodiffContext,
        rng: std.Random,
    ) !u8 {
        // Epsilon-greedy
        if (rng.float(f32) < self.epsilon) {
            // Random action
            return rng.intRangeAtMost(u8, 0, 11);
        }

        // Greedy action from Q-network
        var state_input: [324]f32 = undefined;
        state.toOneHot(&state_input);

        const state_tensor = try tensor_ctx.allocTensor(&[_]usize{ 1, 324 });
        @memcpy(state_tensor.data, &state_input);

        const state_tracked = try ad_ctx.track(state_tensor);

        const q_values = try QNetwork.predict(
            &self.qnet.online_model,
            state_tracked,
            ad_ctx,
            tensor_ctx,
        );

        // Find action with max Q-value
        var max_action: u8 = 0;
        var max_q = q_values.data()[0];
        for (q_values.data()[1..], 1..) |q, i| {
            if (q > max_q) {
                max_q = q;
                max_action = @intCast(i);
            }
        }

        return max_action;
    }

    /// Train on a batch from replay buffer
    pub fn trainStep(
        self: *DQNAgent,
        tensor_ctx: *TensorContext,
        grad_ctx: *GradContext,
        ad_ctx: *AutodiffContext,
        rng: std.Random,
    ) !f32 {
        if (!self.replay_buffer.canSample(self.config.batch_size)) {
            return 0.0; // Not enough samples yet
        }

        const batch_size = self.config.batch_size;

        // Allocate batch tensors
        const states = try self.allocator.alloc(f32, batch_size * 324);
        defer self.allocator.free(states);
        const actions = try self.allocator.alloc(u8, batch_size);
        defer self.allocator.free(actions);
        const rewards = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(rewards);
        const next_states = try self.allocator.alloc(f32, batch_size * 324);
        defer self.allocator.free(next_states);
        const dones = try self.allocator.alloc(bool, batch_size);
        defer self.allocator.free(dones);

        // Sample batch
        try self.replay_buffer.sample(
            batch_size,
            rng,
            states,
            actions,
            rewards,
            next_states,
            dones,
        );

        // 1. Forward pass: Q(s, a) for all actions
        const states_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, 324 });
        @memcpy(states_tensor.data, states);
        const states_tracked = try ad_ctx.track(states_tensor);

        const q_all = try self.qnet.online_model.forward(states_tracked, ad_ctx, tensor_ctx);

        // Extract Q-values for taken actions: Q(s, a)
        const q_sa = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(q_sa);
        try tensor_mod.ops.gatherActions(q_sa, q_all.data(), actions, batch_size, 12);

        // 2. Compute targets: r + gamma * max_a' Q_target(s', a') * (1 - done)
        const next_states_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, 324 });
        @memcpy(next_states_tensor.data, next_states);
        const next_states_tracked = try ad_ctx.track(next_states_tensor);

        const next_q_all = try self.qnet.target_model.forward(next_states_tracked, ad_ctx, tensor_ctx);

        // Get max Q-value for each next state
        const next_q_max = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(next_q_max);
        try tensor_mod.ops.maxAlongAxis1(next_q_max, next_q_all.data(), batch_size, 12);

        // Compute TD targets: r + gamma * max_a' Q(s', a') * (1 - done)
        const targets = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(targets);
        for (0..batch_size) |i| {
            const done_mask: f32 = if (dones[i]) 0.0 else 1.0;
            targets[i] = rewards[i] + self.config.gamma * next_q_max[i] * done_mask;
        }

        // 3. Create tracked tensors for loss computation
        const q_sa_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});
        @memcpy(q_sa_tensor.data, q_sa);
        const q_sa_tracked = try ad_ctx.track(q_sa_tensor);

        const targets_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});
        @memcpy(targets_tensor.data, targets);
        const targets_tracked = try ad_ctx.track(targets_tensor);

        // Allocate tensors for MSE intermediate values
        const diff_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});
        const squared_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});

        // Compute element-wise MSE loss
        const loss_elements = try nn_mod.mse(
            ad_ctx,
            q_sa_tracked,
            targets_tracked,
            diff_tensor,
            squared_tensor,
        );

        // Reduce to scalar loss
        const loss_scalar = tensor_mod.ops.mean(loss_elements.data());

        // 4. Backward pass
        ad_ctx.seedGrad(loss_elements, 1.0);
        ad_ctx.backward();

        // 5. SGD update
        const params = try self.qnet.online_model.getParameterTensors(self.allocator);
        defer self.allocator.free(params);
        const handles = self.qnet.online_model.getParameterHandles();
        nn_mod.sgdStep(params, handles, grad_ctx, self.config.learning_rate);

        // Clear gradients for next iteration
        grad_ctx.zeroGrads();

        return loss_scalar;
    }

    /// Decay epsilon
    pub fn decayEpsilon(self: *DQNAgent) void {
        self.epsilon = @max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay);
    }

    /// Update target network
    pub fn updateTarget(self: *DQNAgent) void {
        self.qnet.updateTargetHard();
    }
};

test "dqn agent init" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const layer_sizes = [_]usize{ 324, 64, 12 };
    const config = DQNConfig{};

    var agent = try DQNAgent.init(
        std.testing.allocator,
        &layer_sizes,
        &tensor_ctx,
        &grad_ctx,
        config,
        rng,
    );
    defer agent.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), agent.epsilon);
    try std.testing.expectEqual(@as(usize, 0), agent.step_count);
}
