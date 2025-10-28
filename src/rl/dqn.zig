const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const nn_mod = @import("../nn.zig");
const env_mod = @import("../env.zig");
const replay_mod = @import("replay.zig");
const prioritized_replay_mod = @import("prioritized_replay.zig");
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

    // N-step returns
    n_step: usize = 3, // N-step horizon for bootstrapping (1 = standard TD, 3-5 = multi-step)

    // Dueling DQN architecture
    use_dueling: bool = true, // Enable dueling architecture (separate value and advantage heads)

    // Prioritized Experience Replay parameters
    use_per: bool = false, // Enable Prioritized Experience Replay
    per_alpha: f32 = 0.6, // Priority exponent (0 = uniform, 1 = full prioritization)
    per_beta_start: f32 = 0.4, // Initial importance sampling exponent
    per_beta_end: f32 = 1.0, // Final importance sampling exponent
    per_beta_frames: usize = 100000, // Number of frames to anneal beta over
    per_epsilon: f32 = 1e-6, // Small constant to ensure non-zero priorities
};

/// Training diagnostics returned from trainStep
pub const TrainingDiagnostics = struct {
    loss: f32,
    td_error_mean: f32,
    td_error_std: f32,
    td_error_max: f32,
    grad_norm: f32,
    // Performance timing (in microseconds)
    time_total_us: f64,
    time_sample_us: f64,
    time_forward_us: f64,
    time_backward_us: f64,
    time_sgd_us: f64,
    // PER statistics (only relevant if use_per=true)
    priority_min: f32 = 0.0,
    priority_max: f32 = 0.0,
    priority_mean: f32 = 0.0,
    beta: f32 = 0.0, // Current importance sampling exponent
};

/// DQN agent (generic over state type and buffer type)
pub fn DQNAgent(comptime StateType: type, comptime use_per: bool) type {
    const ReplayBuffer = if (use_per)
        prioritized_replay_mod.PrioritizedReplayBuffer(StateType)
    else
        replay_mod.ReplayBuffer(StateType);

    return struct {
        qnet: QNetwork,
        replay_buffer: ReplayBuffer,
        config: DQNConfig,
        epsilon: f32,
        beta: f32, // Current importance sampling exponent (for PER)
        step_count: usize,
        allocator: std.mem.Allocator,
        num_actions: usize,
        state_dim: usize,
        /// Cached parameter array to avoid allocation every trainStep
        params_cache: []Tensor,
        /// Parameter gradient context (long-lived, never reset)
        param_grad_ctx: GradContext,
        /// Temporary gradient context (reset after each training step)
        temp_grad_ctx: GradContext,

        const Self = @This();
        const use_prioritized_replay = use_per;

        pub fn init(
            allocator: std.mem.Allocator,
            layer_sizes: []const usize,
            tensor_ctx: *TensorContext,
            config: DQNConfig,
            rng: std.Random,
        ) !Self {
            // Initialize dual gradient contexts
            var param_grad_ctx = GradContext.init(allocator);
            const temp_grad_ctx = GradContext.init(allocator);

            const qnet = try QNetwork.initWithDueling(
                allocator,
                layer_sizes,
                tensor_ctx,
                &param_grad_ctx,
                rng,
                config.use_dueling,
            );

            // Initialize replay buffer (different constructors for uniform vs PER)
            const replay_buffer = if (use_prioritized_replay)
                try ReplayBuffer.init(allocator, config.replay_buffer_size, config.per_alpha, config.per_epsilon)
            else
                try ReplayBuffer.init(allocator, config.replay_buffer_size);

            // Extract dimensions from layer_sizes
            const state_dim = layer_sizes[0];
            const num_actions = layer_sizes[layer_sizes.len - 1];

            // Allocate parameter cache (2 per layer: weights + biases)
            const num_params = qnet.online_model.param_handles.len;
            const params_cache = try allocator.alloc(Tensor, num_params);

            return Self{
                .qnet = qnet,
                .replay_buffer = replay_buffer,
                .config = config,
                .epsilon = config.epsilon_start,
                .beta = config.per_beta_start, // Initial beta for PER
                .step_count = 0,
                .allocator = allocator,
                .num_actions = num_actions,
                .state_dim = state_dim,
                .params_cache = params_cache,
                .param_grad_ctx = param_grad_ctx,
                .temp_grad_ctx = temp_grad_ctx,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.params_cache);
            self.qnet.deinit();
            self.replay_buffer.deinit();
            self.param_grad_ctx.deinit();
            self.temp_grad_ctx.deinit();
        }

        /// Epsilon-greedy action selection
        /// Generic version that works with any state size
        pub fn selectAction(
            self: *Self,
            state_onehot: []const f32,
            tensor_ctx: *TensorContext,
            rng: std.Random,
        ) !u8 {
        std.debug.assert(state_onehot.len == self.state_dim);

        // Epsilon-greedy
        if (rng.float(f32) < self.epsilon) {
            // Random action
            return rng.intRangeAtMost(u8, 0, @intCast(self.num_actions - 1));
        }

        // Greedy action from Q-network (use temp context for inference)
        var ad_ctx = AutodiffContext.init(self.allocator, &self.param_grad_ctx, &self.temp_grad_ctx);
        defer ad_ctx.deinit();

        const state_tensor = try tensor_ctx.allocTensor(&[_]usize{ 1, self.state_dim });
        @memcpy(state_tensor.data, state_onehot);

        const state_tracked = try ad_ctx.track(state_tensor);

        const q_values = try QNetwork.predict(
            &self.qnet.online_model,
            state_tracked,
            &ad_ctx,
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

        // Reset temp context after inference
        self.temp_grad_ctx.reset();

        return max_action;
    }

        /// Train on a batch from replay buffer
        pub fn trainStep(
            self: *Self,
            tensor_ctx: *TensorContext,
            rng: std.Random,
        ) !TrainingDiagnostics {
        if (!self.replay_buffer.canSample(self.config.batch_size)) {
            return TrainingDiagnostics{
                .loss = 0.0,
                .td_error_mean = 0.0,
                .td_error_std = 0.0,
                .td_error_max = 0.0,
                .grad_norm = 0.0,
                .time_total_us = 0.0,
                .time_sample_us = 0.0,
                .time_forward_us = 0.0,
                .time_backward_us = 0.0,
                .time_sgd_us = 0.0,
            };
        }

        // Create autodiff context with dual gradient contexts
        var ad_ctx = AutodiffContext.init(self.allocator, &self.param_grad_ctx, &self.temp_grad_ctx);
        defer ad_ctx.deinit();

        const batch_size = self.config.batch_size;

        // Timing instrumentation
        const t_start = std.time.nanoTimestamp();
        var t_sample: i128 = 0;
        var t_forward: i128 = 0;
        var t_backward: i128 = 0;
        var t_sgd: i128 = 0;

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
        const n_steps_taken = try self.allocator.alloc(u8, batch_size);
        defer self.allocator.free(n_steps_taken);

        // PER-specific arrays (only used if use_prioritized_replay)
        const indices = try self.allocator.alloc(usize, batch_size);
        defer self.allocator.free(indices);
        const weights = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(weights);
        const td_errors = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(td_errors);

        // Sample batch (comptime dispatch for zero overhead)
        const t_sample_start = std.time.nanoTimestamp();
        if (use_prioritized_replay) {
            // Update beta (annealing schedule)
            const progress = @min(1.0, @as(f32, @floatFromInt(self.step_count)) /
                @as(f32, @floatFromInt(self.config.per_beta_frames)));
            self.beta = self.config.per_beta_start + progress * (self.config.per_beta_end - self.config.per_beta_start);

            // Prioritized sampling with importance weights
            try self.replay_buffer.sample(
                batch_size,
                self.beta,
                rng,
                states,
                actions,
                rewards,
                next_states,
                dones,
                n_steps_taken,
                indices,
                weights,
                self.state_dim,
            );
        } else {
            // Uniform sampling (standard replay)
            try self.replay_buffer.sample(
                batch_size,
                rng,
                states,
                actions,
                rewards,
                next_states,
                dones,
                n_steps_taken,
                self.state_dim,
            );
            // Set uniform weights
            @memset(weights, 1.0);
        }
        t_sample = std.time.nanoTimestamp() - t_sample_start;

        // 1. Forward pass: Q(s, a) for all actions
        const t_forward_start = std.time.nanoTimestamp();
        const states_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.state_dim });
        @memcpy(states_tensor.data, states);
        const states_tracked = try ad_ctx.track(states_tensor);

        const q_all = try self.qnet.online_model.forward(states_tracked, &ad_ctx, tensor_ctx);

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
            const next_q_online = try self.qnet.online_model.forward(next_states_tracked, &ad_ctx, tensor_ctx);
            const next_q_target = try self.qnet.target_model.forward(next_states_tracked, &ad_ctx, tensor_ctx);

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
            const next_q_all = try self.qnet.target_model.forward(next_states_tracked, &ad_ctx, tensor_ctx);
            try tensor_mod.ops.maxAlongAxis1(next_q_max, next_q_all.data(), batch_size, self.num_actions);
        }

        // Compute TD targets: r + gamma^n * max_a' Q(s', a') * (1 - done)
        // where n = n_steps_taken (to account for n-step returns)
        const targets = try self.allocator.alloc(f32, batch_size);
        defer self.allocator.free(targets);
        for (0..batch_size) |i| {
            const done_mask: f32 = if (dones[i]) 0.0 else 1.0;
            // Use gamma^n_steps_taken for n-step bootstrapping
            const n = @as(f32, @floatFromInt(n_steps_taken[i]));
            const gamma_n = std.math.pow(f32, self.config.gamma, n);
            targets[i] = rewards[i] + gamma_n * next_q_max[i] * done_mask;
        }

        // Compute TD-error stats for diagnostics: td_error = targets - q_sa
        // Also store TD errors for PER priority updates
        const q_sa_data = q_sa_tracked.data();
        var td_error_sum: f32 = 0.0;
        var td_error_sum_sq: f32 = 0.0;
        var td_error_max: f32 = 0.0;
        for (0..batch_size) |i| {
            const td_error = targets[i] - q_sa_data[i];
            td_errors[i] = td_error; // Store for priority updates
            td_error_sum += td_error;
            td_error_sum_sq += td_error * td_error;
            td_error_max = @max(td_error_max, @abs(td_error));
        }
        const td_error_mean = td_error_sum / @as(f32, @floatFromInt(batch_size));
        const td_error_variance = (td_error_sum_sq / @as(f32, @floatFromInt(batch_size))) - (td_error_mean * td_error_mean);
        const td_error_std = @sqrt(@max(0.0, td_error_variance));

        t_forward = std.time.nanoTimestamp() - t_forward_start;

        // 3. Create tracked tensor for targets
        const targets_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});
        @memcpy(targets_tensor.data, targets);
        const targets_tracked = try ad_ctx.track(targets_tensor);

        // Allocate tensors for MSE intermediate values
        const diff_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});
        const squared_tensor = try tensor_ctx.allocTensor(&[_]usize{batch_size});

        // Compute element-wise MSE loss
        const loss_elements = try nn_mod.mse(
            &ad_ctx,
            q_sa_tracked,
            targets_tracked,
            diff_tensor,
            squared_tensor,
        );

        // Weight loss elements by importance sampling weights (PER)
        // For uniform replay, weights are all 1.0, so this is a no-op
        const loss_data = loss_elements.data();
        for (0..batch_size) |i| {
            loss_data[i] *= weights[i];
        }

        // Reduce to scalar loss
        const loss_scalar = tensor_mod.ops.mean(loss_data);

        // 4. Backward pass: seed with 1/N for mean reduction
        const t_backward_start = std.time.nanoTimestamp();
        const grad_scale = 1.0 / @as(f32, @floatFromInt(loss_elements.data().len));
        ad_ctx.seedGrad(loss_elements, grad_scale);
        ad_ctx.backward();
        t_backward = std.time.nanoTimestamp() - t_backward_start;

        // 5. SGD update
        const t_sgd_start = std.time.nanoTimestamp();

        // (a) Populate parameter cache (avoids allocation)
        const t_cache_fill_start = std.time.nanoTimestamp();

        // Backbone parameters
        for (self.qnet.online_model.layers, 0..) |layer, i| {
            self.params_cache[i * 2] = layer.W;
            self.params_cache[i * 2 + 1] = layer.b;
        }

        // Dueling heads parameters if present
        if (self.qnet.online_model.use_dueling) {
            const value_offset = self.qnet.online_model.layers.len * 2;
            if (self.qnet.online_model.value_head) |vh| {
                self.params_cache[value_offset] = vh.W;
                self.params_cache[value_offset + 1] = vh.b;
            }
            if (self.qnet.online_model.advantage_head) |ah| {
                const adv_offset = value_offset + 2;
                self.params_cache[adv_offset] = ah.W;
                self.params_cache[adv_offset + 1] = ah.b;
            }
        }

        const t_cache_fill = std.time.nanoTimestamp() - t_cache_fill_start;

        const handles = self.qnet.online_model.getParameterHandles();

        // Compute gradient norm for diagnostics (first layer)
        const grad_first_layer = self.param_grad_ctx.getGrad(handles[0]);
        var grad_norm_sq: f32 = 0.0;
        for (grad_first_layer) |g| {
            grad_norm_sq += g * g;
        }
        const grad_norm = @sqrt(grad_norm_sq);

        // Diagnostic: log first layer weight norm before/after SGD (first 10 steps only)
        if (self.step_count < 10) {
            var norm_before: f32 = 0.0;
            for (self.params_cache[0].data) |w| {
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

            // (b) Time SGD update
            const t_sgd_update_start = std.time.nanoTimestamp();
            nn_mod.sgdStepClipped(self.params_cache, handles, &self.param_grad_ctx, self.config.learning_rate, 1.0);
            const t_sgd_update = std.time.nanoTimestamp() - t_sgd_update_start;

            var norm_after: f32 = 0.0;
            for (self.params_cache[0].data) |w| {
                norm_after += w * w;
            }
            norm_after = @sqrt(norm_after);

            std.debug.print("  [PARAM NORM] Step {d}: before={d:.6}, after={d:.6}, delta={d:.6}\n", .{
                self.step_count, norm_before, norm_after, norm_after - norm_before
            });
            std.debug.print("  [SGD TIMING] Step {d}: cache_fill={d}µs, sgd_update={d}µs\n", .{
                self.step_count,
                @as(f64, @floatFromInt(t_cache_fill)) / 1000.0,
                @as(f64, @floatFromInt(t_sgd_update)) / 1000.0,
            });
        } else {
            // (b) Time SGD update
            const t_sgd_update_start = std.time.nanoTimestamp();
            nn_mod.sgdStepClipped(self.params_cache, handles, &self.param_grad_ctx, self.config.learning_rate, 1.0);
            const t_sgd_update = std.time.nanoTimestamp() - t_sgd_update_start;

            // Log detailed timing every 100 steps after warmup
            if (self.step_count % 100 == 0) {
                std.debug.print("  [SGD TIMING] Step {d}: cache_fill={d:.1}µs, sgd_update={d:.1}µs\n", .{
                    self.step_count,
                    @as(f64, @floatFromInt(t_cache_fill)) / 1000.0,
                    @as(f64, @floatFromInt(t_sgd_update)) / 1000.0,
                });
            }
        }

        self.step_count += 1;

        // (c) Zero parameter gradients for next iteration
        const t_zero_start = std.time.nanoTimestamp();
        self.param_grad_ctx.zeroGrads();
        const t_zero = std.time.nanoTimestamp() - t_zero_start;

        // Reset temporary gradients (frees intermediate memory)
        const t_temp_reset_start = std.time.nanoTimestamp();
        self.temp_grad_ctx.reset();
        const t_temp_reset = std.time.nanoTimestamp() - t_temp_reset_start;

        // Log detailed timing breakdown every 100 steps
        if (self.step_count % 100 == 0 and self.step_count > 10) {
            const param_bytes = self.param_grad_ctx.getTotalBytes();
            std.debug.print("  [SGD BREAKDOWN] Step {d}: zero_grads={d:.1}µs ({d} KB), temp_reset={d:.1}µs\n", .{
                self.step_count,
                @as(f64, @floatFromInt(t_zero)) / 1000.0,
                param_bytes / 1024,
                @as(f64, @floatFromInt(t_temp_reset)) / 1000.0,
            });
        }

        t_sgd = std.time.nanoTimestamp() - t_sgd_start;
        const t_total = std.time.nanoTimestamp() - t_start;

        // Update priorities in PER buffer based on TD errors
        if (use_prioritized_replay) {
            self.replay_buffer.updatePriorities(indices, td_errors);
        }

        // Get priority statistics for diagnostics
        const priority_stats = if (use_prioritized_replay)
            self.replay_buffer.getPriorityStats()
        else
            .{ .min = 0.0, .max = 0.0, .mean = 0.0, .total = 0.0 };

        // Convert to microseconds for readability
        const ns_to_us = 1000.0;

        return TrainingDiagnostics{
            .loss = loss_scalar,
            .td_error_mean = td_error_mean,
            .td_error_std = td_error_std,
            .td_error_max = td_error_max,
            .grad_norm = grad_norm,
            .time_total_us = @as(f64, @floatFromInt(t_total)) / ns_to_us,
            .time_sample_us = @as(f64, @floatFromInt(t_sample)) / ns_to_us,
            .time_forward_us = @as(f64, @floatFromInt(t_forward)) / ns_to_us,
            .time_backward_us = @as(f64, @floatFromInt(t_backward)) / ns_to_us,
            .time_sgd_us = @as(f64, @floatFromInt(t_sgd)) / ns_to_us,
            .priority_min = priority_stats.min,
            .priority_max = priority_stats.max,
            .priority_mean = priority_stats.mean,
            .beta = self.beta,
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
        ) !f32 {
            // Create autodiff context
            var ad_ctx = AutodiffContext.init(self.allocator, &self.param_grad_ctx, &self.temp_grad_ctx);
            defer ad_ctx.deinit();

            // Forward pass
            const states_tensor = try tensor_ctx.allocTensor(&[_]usize{ batch_size, self.state_dim });
            @memcpy(states_tensor.data, states);
            const states_tracked = try ad_ctx.track(states_tensor);

            const q_all = try self.qnet.online_model.forward(states_tracked, &ad_ctx, tensor_ctx);

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
                &ad_ctx,
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
            nn_mod.sgdStepClipped(params, handles, &self.param_grad_ctx, self.config.learning_rate, 1.0);

            // Zero parameter gradients
            self.param_grad_ctx.zeroGrads();

            // Reset temporary gradients
            self.temp_grad_ctx.reset();

            return loss_scalar;
        }
    };
}

// Convenience type aliases for 2x2 Cube DQN
pub const DQN = DQNAgent(CubeState, false);
pub const DQN_PER = DQNAgent(CubeState, true);

test "dqn agent init" {
    var tensor_ctx = TensorContext.init(std.testing.allocator);
    defer tensor_ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const layer_sizes = [_]usize{ 324, 64, 12 };
    const config = DQNConfig{};

    var agent = try DQN.init(
        std.testing.allocator,
        &layer_sizes,
        &tensor_ctx,
        config,
        rng,
    );
    defer agent.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), agent.epsilon);
    try std.testing.expectEqual(@as(usize, 0), agent.step_count);
}
