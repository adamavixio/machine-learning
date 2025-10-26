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
    use_double_dqn: bool = true, // Use Double DQN to reduce overestimation
    polyak_tau: f32 = 0.05, // Polyak averaging coefficient (0 = no update, 1 = hard update)
};

/// Training diagnostics returned from trainStep
pub const TrainingDiagnostics = struct {
    loss: f32,
    td_error_mean: f32,
    td_error_std: f32,
    td_error_max: f32,
    grad_norm: f32,
};

/// DQN agent (generic over state type)
pub fn DQNAgent(comptime StateType: type) type {
    const ReplayBuffer = replay_mod.ReplayBuffer(StateType);

    return struct {
        qnet: QNetwork,
        replay_buffer: ReplayBuffer,
        config: DQNConfig,
        epsilon: f32,
        step_count: usize,
        allocator: std.mem.Allocator,
        num_actions: usize,
        state_dim: usize,

        const Self = @This();

        pub fn init(
            allocator: std.mem.Allocator,
            layer_sizes: []const usize,
            tensor_ctx: *TensorContext,
            grad_ctx: *GradContext,
            config: DQNConfig,
            rng: std.Random,
        ) !Self {
            const qnet = try QNetwork.init(
                allocator,
                layer_sizes,
                tensor_ctx,
                grad_ctx,
                rng,
            );

            const replay_buffer = try ReplayBuffer.init(allocator, config.replay_buffer_size);

            // Extract dimensions from layer_sizes
            const state_dim = layer_sizes[0];
            const num_actions = layer_sizes[layer_sizes.len - 1];

            return Self{
                .qnet = qnet,
                .replay_buffer = replay_buffer,
                .config = config,
                .epsilon = config.epsilon_start,
                .step_count = 0,
                .allocator = allocator,
                .num_actions = num_actions,
                .state_dim = state_dim,
            };
        }

        pub fn deinit(self: *Self) void {
            self.qnet.deinit();
            self.replay_buffer.deinit();
        }

        /// Epsilon-greedy action selection
        /// Generic version that works with any state size
        pub fn selectAction(
            self: *Self,
            state_onehot: []const f32,
            tensor_ctx: *TensorContext,
            ad_ctx: *AutodiffContext,
            rng: std.Random,
        ) !u8 {
        std.debug.assert(state_onehot.len == self.state_dim);

        // Epsilon-greedy
        if (rng.float(f32) < self.epsilon) {
            // Random action
            return rng.intRangeAtMost(u8, 0, @intCast(self.num_actions - 1));
        }

        // Greedy action from Q-network
        const state_tensor = try tensor_ctx.allocTensor(&[_]usize{ 1, self.state_dim });
        @memcpy(state_tensor.data, state_onehot);

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
            self: *Self,
            tensor_ctx: *TensorContext,
            grad_ctx: *GradContext,
            ad_ctx: *AutodiffContext,
            rng: std.Random,
        ) !TrainingDiagnostics {
        if (!self.replay_buffer.canSample(self.config.batch_size)) {
            return TrainingDiagnostics{
                .loss = 0.0,
                .td_error_mean = 0.0,
                .td_error_std = 0.0,
                .td_error_max = 0.0,
                .grad_norm = 0.0,
            };
        }

        const batch_size = self.config.batch_size;

        // Allocate batch tensors
        const states = try self.allocator.alloc(f32, batch_size * self.state_dim);
        defer self.allocator.free(states);
        const actions = try self.allocator.alloc(u8, batch_size);
        defer self.allocator.free(actions);
        const rewards = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(rewards);
        const next_states = try self.allocator.alloc(f32, batch_size * self.state_dim);
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
            self.state_dim,
        );

        // 1. Forward pass: Q(s, a) for all actions
        const states_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.state_dim });
        @memcpy(states_tensor.data, states);
        const states_tracked = try ad_ctx.track(states_tensor);

        const q_all = try self.qnet.online_model.forward(states_tracked, ad_ctx, tensor_ctx);

        // Extract Q-values for taken actions using tracked gather: Q(s, a)
        const q_sa_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});
        const q_sa_tracked = try ad_ctx.trackedGatherActions(
            q_all,
            actions,
            batch_size,
            self.num_actions,
            q_sa_tensor,
        );

        // 2. Compute targets using Double DQN or vanilla DQN
        const next_states_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.state_dim });
        @memcpy(next_states_tensor.data, next_states);
        const next_states_tracked = try ad_ctx.track(next_states_tensor);

        const next_q_max = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(next_q_max);

        if (self.config.use_double_dqn) {
            // Double DQN: use online network to select action, target network to evaluate
            const next_q_online = try self.qnet.online_model.forward(next_states_tracked, ad_ctx, tensor_ctx);
            const next_q_target = try self.qnet.target_model.forward(next_states_tracked, ad_ctx, tensor_ctx);

            // Select best actions using online network
            const best_actions = try self.allocator.alloc(u8, batch_size);
            defer self.allocator.free(best_actions);

            for (0..batch_size) |i| {
                var max_q: f32 = next_q_online.data()[i * self.num_actions];
                var best_action: u8 = 0;
                for (1..self.num_actions) |a| {
                    const q_val = next_q_online.data()[i * self.num_actions + a];
                    if (q_val > max_q) {
                        max_q = q_val;
                        best_action = @intCast(a);
                    }
                }
                best_actions[i] = best_action;
            }

            // Evaluate those actions using target network
            try tensor_mod.ops.gatherActions(next_q_max, next_q_target.data(), best_actions, batch_size, self.num_actions);
        } else {
            // Vanilla DQN: max over target network
            const next_q_all = try self.qnet.target_model.forward(next_states_tracked, ad_ctx, tensor_ctx);
            try tensor_mod.ops.maxAlongAxis1(next_q_max, next_q_all.data(), batch_size, self.num_actions);
        }

        // Compute TD targets: r + gamma * max_a' Q(s', a') * (1 - done)
        const targets = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(targets);
        for (0..batch_size) |i| {
            const done_mask: f32 = if (dones[i]) 0.0 else 1.0;
            targets[i] = rewards[i] + self.config.gamma * next_q_max[i] * done_mask;
        }

        // Compute TD-error stats for diagnostics: td_error = targets - q_sa
        const q_sa_data = q_sa_tracked.data();
        var td_error_sum: f32 = 0.0;
        var td_error_sum_sq: f32 = 0.0;
        var td_error_max: f32 = 0.0;
        for (0..batch_size) |i| {
            const td_error = targets[i] - q_sa_data[i];
            td_error_sum += td_error;
            td_error_sum_sq += td_error * td_error;
            td_error_max = @max(td_error_max, @abs(td_error));
        }
        const td_error_mean = td_error_sum / @as(f32, @floatFromInt(batch_size));
        const td_error_variance = (td_error_sum_sq / @as(f32, @floatFromInt(batch_size))) - (td_error_mean * td_error_mean);
        const td_error_std = @sqrt(@max(0.0, td_error_variance));

        // 3. Create tracked tensor for targets
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

        // 4. Backward pass: seed with 1/N for mean reduction
        const grad_scale = 1.0 / @as(f32, @floatFromInt(loss_elements.data().len));
        ad_ctx.seedGrad(loss_elements, grad_scale);
        ad_ctx.backward();

        // 5. SGD update
        const params = try self.qnet.online_model.getParameterTensors(self.allocator);
        defer self.allocator.free(params);
        const handles = self.qnet.online_model.getParameterHandles();

        // Compute gradient norm for diagnostics (first layer)
        const grad_first_layer = grad_ctx.getGrad(handles[0]);
        var grad_norm_sq: f32 = 0.0;
        for (grad_first_layer) |g| {
            grad_norm_sq += g * g;
        }
        const grad_norm = @sqrt(grad_norm_sq);

        // Diagnostic: log first layer weight norm before/after SGD (first 10 steps only)
        if (self.step_count < 10) {
            var norm_before: f32 = 0.0;
            for (params[0].data) |w| {
                norm_before += w * w;
            }
            norm_before = @sqrt(norm_before);

            // Check gradient values (reuse grad_first_layer from above)
            var grad_sum: f32 = 0.0;
            var grad_nonzero_count: usize = 0;
            for (grad_first_layer) |g| {
                grad_sum += g;
                if (g != 0.0) grad_nonzero_count += 1;
            }

            std.debug.print("  [GRAD CHECK] Step {d}: grad_norm={d:.6}, grad_mean={d:.6}, nonzero={d}/{d}\n", .{
                self.step_count, grad_norm, grad_sum / @as(f32, @floatFromInt(grad_first_layer.len)), grad_nonzero_count, grad_first_layer.len
            });

            nn_mod.sgdStepClipped(params, handles, grad_ctx, self.config.learning_rate, 1.0);

            var norm_after: f32 = 0.0;
            for (params[0].data) |w| {
                norm_after += w * w;
            }
            norm_after = @sqrt(norm_after);

            std.debug.print("  [PARAM NORM] Step {d}: before={d:.6}, after={d:.6}, delta={d:.6}\n", .{
                self.step_count, norm_before, norm_after, norm_after - norm_before
            });
        } else {
            nn_mod.sgdStepClipped(params, handles, grad_ctx, self.config.learning_rate, 1.0);
        }

        self.step_count += 1;

        // Clear gradients for next iteration
        grad_ctx.zeroGrads();

        return TrainingDiagnostics{
            .loss = loss_scalar,
            .td_error_mean = td_error_mean,
            .td_error_std = td_error_std,
            .td_error_max = td_error_max,
            .grad_norm = grad_norm,
        };
    }

        /// Decay epsilon
        pub fn decayEpsilon(self: *Self) void {
            self.epsilon = @max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay);
        }

        /// Update target network using Polyak averaging
        pub fn updateTarget(self: *Self) void {
            if (self.config.polyak_tau >= 1.0) {
                self.qnet.updateTargetHard();
            } else {
                self.qnet.updateTargetPolyak(self.config.polyak_tau);
            }
        }

        /// Compute L2 distance between online and target network parameters
        pub fn computeTargetDrift(self: *Self) !f32 {
            const online_params = try self.qnet.online_model.getParameterTensors(self.allocator);
            defer self.allocator.free(online_params);
            const target_params = try self.qnet.target_model.getParameterTensors(self.allocator);
            defer self.allocator.free(target_params);

            var l2_dist: f32 = 0.0;
            for (online_params, target_params) |op, tp| {
                for (op.data, tp.data) |ov, tv| {
                    const diff = ov - tv;
                    l2_dist += diff * diff;
                }
            }
            return @sqrt(l2_dist);
        }

        /// Supervised pretraining on (state, action) pairs
        /// Returns average loss over the batch
        pub fn pretrainStep(
            self: *Self,
            states: []const f32,  // [batch_size * state_dim]
            actions: []const u8,  // [batch_size]
            batch_size: usize,
            tensor_ctx: *TensorContext,
            grad_ctx: *GradContext,
            ad_ctx: *AutodiffContext,
        ) !f32 {
            // Forward pass
            const states_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.state_dim });
            @memcpy(states_tensor.data, states);
            const states_tracked = try ad_ctx.track(states_tensor);

            const q_all = try self.qnet.online_model.forward(states_tracked, ad_ctx, tensor_ctx);

            // Create targets: high Q-value (+10) for correct action, low (-10) for others
            const targets = try self.allocator.alloc(f32, batch_size * self.num_actions);
            defer self.allocator.free(targets);
            @memset(targets, -10.0);
            for (actions, 0..) |action, i| {
                targets[i * self.num_actions + action] = 10.0;
            }

            const targets_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.num_actions });
            @memcpy(targets_tensor.data, targets);
            const targets_tracked = try ad_ctx.track(targets_tensor);

            // MSE loss between Q-values and one-hot targets
            const diff_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.num_actions });
            const squared_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.num_actions });

            const loss_elements = try nn_mod.mse(
                ad_ctx,
                q_all,
                targets_tracked,
                diff_tensor,
                squared_tensor,
            );

            const loss_scalar = tensor_mod.ops.mean(loss_elements.data());

            // Backward pass: seed with 1/N for mean reduction
            const grad_scale = 1.0 / @as(f32, @floatFromInt(loss_elements.data().len));
            ad_ctx.seedGrad(loss_elements, grad_scale);
            ad_ctx.backward();

            // SGD update with gradient clipping
            const params = try self.qnet.online_model.getParameterTensors(self.allocator);
            defer self.allocator.free(params);
            const handles = self.qnet.online_model.getParameterHandles();
            nn_mod.sgdStepClipped(params, handles, grad_ctx, self.config.learning_rate, 1.0);

            // Clear gradients
            grad_ctx.zeroGrads();

            return loss_scalar;
        }
    };
}

test "dqn agent init" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const layer_sizes = [_]usize{ 324, 64, 12 };
    const config = DQNConfig{};

    var agent = try DQNAgent(CubeState).init(
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
