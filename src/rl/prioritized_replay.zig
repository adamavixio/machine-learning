const std = @import("std");

/// Sum tree for efficient prioritized sampling
/// Binary tree where each parent node contains the sum of its children
/// Leaf nodes contain experience priorities
pub const SumTree = struct {
    /// Tree array: [internal nodes | leaf nodes]
    /// Size: 2 * capacity - 1
    /// First (capacity - 1) elements are internal nodes
    /// Last capacity elements are leaf nodes (priorities)
    tree: []f32,
    capacity: usize,
    write_pos: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !SumTree {
        // Binary tree size: 2n - 1
        const tree_size = 2 * capacity - 1;
        const tree = try allocator.alloc(f32, tree_size);
        @memset(tree, 0.0);

        return SumTree{
            .tree = tree,
            .capacity = capacity,
            .write_pos = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SumTree) void {
        self.allocator.free(self.tree);
    }

    /// Update priority at leaf index and propagate changes up the tree
    pub fn update(self: *SumTree, data_idx: usize, priority: f32) void {
        std.debug.assert(data_idx < self.capacity);

        // Leaf nodes start at (capacity - 1)
        const tree_idx = data_idx + self.capacity - 1;

        // Calculate the change in priority
        const change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;

        // Propagate change up the tree
        self.propagate(tree_idx, change);
    }

    /// Propagate priority change up the tree
    fn propagate(self: *SumTree, idx: usize, change: f32) void {
        var current_idx = idx;

        // Traverse up to root (idx 0)
        while (current_idx != 0) {
            // Parent index: (i - 1) / 2
            current_idx = (current_idx - 1) / 2;
            self.tree[current_idx] += change;
        }
    }

    /// Sample a leaf index proportional to priority
    /// Returns the data index (0 to capacity-1) and the priority value
    pub fn sample(self: *SumTree, value: f32) struct { data_idx: usize, priority: f32 } {
        std.debug.assert(value >= 0.0 and value <= self.total());

        const leaf_idx = self.retrieve(0, value);
        const data_idx = leaf_idx - (self.capacity - 1);

        return .{
            .data_idx = data_idx,
            .priority = self.tree[leaf_idx],
        };
    }

    /// Retrieve leaf index for a given cumulative value
    /// Traverses tree from root, going left/right based on cumulative sums
    fn retrieve(self: *SumTree, idx: usize, value: f32) usize {
        // If we're at a leaf node, return it
        const left_child_idx = 2 * idx + 1;
        if (left_child_idx >= self.tree.len) {
            return idx;
        }

        const left_value = self.tree[left_child_idx];

        // If value is in left subtree, go left
        if (value <= left_value) {
            return self.retrieve(left_child_idx, value);
        } else {
            // Otherwise go right, subtracting left subtree's sum
            const right_child_idx = left_child_idx + 1;
            return self.retrieve(right_child_idx, value - left_value);
        }
    }

    /// Get total sum of all priorities (root value)
    pub fn total(self: *SumTree) f32 {
        return self.tree[0];
    }

    /// Get the next write position and advance
    pub fn nextWritePos(self: *SumTree) usize {
        const pos = self.write_pos;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        return pos;
    }

    /// Get priority at data index
    pub fn getPriority(self: *SumTree, data_idx: usize) f32 {
        std.debug.assert(data_idx < self.capacity);
        return self.tree[data_idx + self.capacity - 1];
    }
};

/// Experience with priority for prioritized replay
pub fn PrioritizedExperience(comptime StateType: type) type {
    return struct {
        state: StateType,
        action: u8,
        reward: f32, // Pre-computed n-step return
        next_state: StateType, // State after n steps
        done: bool,
        episode_id: usize = 0,
        n_steps_taken: u8 = 1, // Actual number of steps taken (1 to n_step)
    };
}

/// Prioritized replay buffer using sum tree for efficient sampling
pub fn PrioritizedReplayBuffer(comptime StateType: type) type {
    const Exp = PrioritizedExperience(StateType);

    return struct {
        data: []Exp,
        sum_tree: SumTree,
        capacity: usize,
        size: usize,
        allocator: std.mem.Allocator,

        // Hyperparameters
        alpha: f32, // Priority exponent (0 = uniform, 1 = full prioritization)
        epsilon: f32, // Small constant to ensure non-zero priorities

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, capacity: usize, alpha: f32, epsilon: f32) !Self {
            const data = try allocator.alloc(Exp, capacity);
            const sum_tree = try SumTree.init(allocator, capacity);

            return Self{
                .data = data,
                .sum_tree = sum_tree,
                .capacity = capacity,
                .size = 0,
                .allocator = allocator,
                .alpha = alpha,
                .epsilon = epsilon,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.sum_tree.deinit();
        }

        /// Add experience with maximum priority (will be updated on first train step)
        pub fn push(self: *Self, exp: Exp) void {
            const idx = self.sum_tree.nextWritePos();
            self.data[idx] = exp;

            // Assign maximum priority to new experiences
            const max_priority = self.getMaxPriority();
            self.sum_tree.update(idx, max_priority);

            if (self.size < self.capacity) {
                self.size += 1;
            }
        }

        /// Get maximum priority in buffer (for new experiences)
        fn getMaxPriority(self: *Self) f32 {
            var max_priority: f32 = 1.0; // Default for empty buffer

            for (0..self.size) |i| {
                const priority = self.sum_tree.getPriority(i);
                max_priority = @max(max_priority, priority);
            }

            return max_priority;
        }

        /// Sample batch with prioritized sampling
        /// Returns sampled experiences, indices, and importance sampling weights
        pub fn sample(
            self: *Self,
            batch_size: usize,
            beta: f32, // Importance sampling exponent
            rng: std.Random,
            states: []f32,
            actions: []u8,
            rewards: []f32,
            next_states: []f32,
            dones: []bool,
            n_steps_taken: []u8,
            indices: []usize,
            weights: []f32,
            state_dim: usize,
        ) !void {
            if (self.size < batch_size) {
                return error.InsufficientSamples;
            }

            const total_priority = self.sum_tree.total();
            const segment_size = total_priority / @as(f32, @floatFromInt(batch_size));

            // Compute minimum probability for importance sampling normalization
            var min_prob = std.math.floatMax(f32);

            // Sample batch
            for (0..batch_size) |i| {
                // Stratified sampling: divide priority range into segments
                const segment_start = segment_size * @as(f32, @floatFromInt(i));
                const segment_end = segment_start + segment_size;
                const value = segment_start + rng.float(f32) * (segment_end - segment_start);

                // Sample from sum tree
                const sample_result = self.sum_tree.sample(value);
                const data_idx = sample_result.data_idx;
                const priority = sample_result.priority;

                // Store index for priority updates
                indices[i] = data_idx;

                // Calculate sampling probability
                const prob = priority / total_priority;
                min_prob = @min(min_prob, prob);

                // Convert experience to batch arrays
                const exp = &self.data[data_idx];

                // One-hot encode states
                const state_offset = i * state_dim;
                exp.state.toOneHot(states[state_offset .. state_offset + state_dim]);
                exp.next_state.toOneHot(next_states[state_offset .. state_offset + state_dim]);

                actions[i] = exp.action;
                rewards[i] = exp.reward;
                dones[i] = exp.done;
                n_steps_taken[i] = exp.n_steps_taken;

                // Store probability for weight calculation (computed below)
                weights[i] = prob;
            }

            // Calculate importance sampling weights
            // Weight = (N * P(i))^(-beta) / max_weight
            const n = @as(f32, @floatFromInt(self.size));
            const max_weight = std.math.pow(f32, n * min_prob, -beta);

            for (0..batch_size) |i| {
                const prob = weights[i];
                const weight = std.math.pow(f32, n * prob, -beta);
                weights[i] = weight / max_weight; // Normalize by max weight
            }
        }

        /// Update priorities based on TD errors
        pub fn updatePriorities(self: *Self, indices: []const usize, td_errors: []const f32) void {
            std.debug.assert(indices.len == td_errors.len);

            for (indices, td_errors) |idx, td_error| {
                // Priority = (|TD error| + epsilon)^alpha
                const priority = std.math.pow(
                    f32,
                    @abs(td_error) + self.epsilon,
                    self.alpha,
                );
                self.sum_tree.update(idx, priority);
            }
        }

        /// Check if buffer has enough samples
        pub fn canSample(self: *Self, batch_size: usize) bool {
            return self.size >= batch_size;
        }

        /// Get current buffer size
        pub fn len(self: *Self) usize {
            return self.size;
        }

        /// Get priority distribution statistics for diagnostics
        pub fn getPriorityStats(self: *Self) struct {
            min: f32,
            max: f32,
            mean: f32,
            total: f32,
        } {
            if (self.size == 0) {
                return .{ .min = 0.0, .max = 0.0, .mean = 0.0, .total = 0.0 };
            }

            var min_p = self.sum_tree.getPriority(0);
            var max_p = min_p;
            var sum_p: f32 = 0.0;

            for (0..self.size) |i| {
                const p = self.sum_tree.getPriority(i);
                min_p = @min(min_p, p);
                max_p = @max(max_p, p);
                sum_p += p;
            }

            return .{
                .min = min_p,
                .max = max_p,
                .mean = sum_p / @as(f32, @floatFromInt(self.size)),
                .total = self.sum_tree.total(),
            };
        }
    };
}

// Tests
test "sum tree init and total" {
    var tree = try SumTree.init(std.testing.allocator, 4);
    defer tree.deinit();

    try std.testing.expectEqual(@as(f32, 0.0), tree.total());
    try std.testing.expectEqual(@as(usize, 4), tree.capacity);
}

test "sum tree update and propagate" {
    var tree = try SumTree.init(std.testing.allocator, 4);
    defer tree.deinit();

    // Update leaf priorities
    tree.update(0, 1.0);
    tree.update(1, 2.0);
    tree.update(2, 3.0);
    tree.update(3, 4.0);

    // Total should be sum of all priorities
    try std.testing.expectEqual(@as(f32, 10.0), tree.total());

    // Update one priority
    tree.update(1, 5.0); // Changed from 2.0 to 5.0
    try std.testing.expectEqual(@as(f32, 13.0), tree.total());
}

test "sum tree sampling" {
    var tree = try SumTree.init(std.testing.allocator, 4);
    defer tree.deinit();

    tree.update(0, 1.0); // Range [0.0, 1.0)
    tree.update(1, 2.0); // Range [1.0, 3.0)
    tree.update(2, 3.0); // Range [3.0, 6.0)
    tree.update(3, 4.0); // Range [6.0, 10.0)

    // Sample from different ranges
    const sample1 = tree.sample(0.5); // Should hit index 0
    try std.testing.expectEqual(@as(usize, 0), sample1.data_idx);

    const sample2 = tree.sample(1.5); // Should hit index 1
    try std.testing.expectEqual(@as(usize, 1), sample2.data_idx);

    const sample3 = tree.sample(4.0); // Should hit index 2
    try std.testing.expectEqual(@as(usize, 2), sample3.data_idx);

    const sample4 = tree.sample(8.0); // Should hit index 3
    try std.testing.expectEqual(@as(usize, 3), sample4.data_idx);
}

test "prioritized replay buffer push and sample" {
    const env_mod = @import("../env.zig");
    const CubeState = env_mod.CubeState;

    var buffer = try PrioritizedReplayBuffer(CubeState).init(
        std.testing.allocator,
        10,
        0.6, // alpha
        1e-6, // epsilon
    );
    defer buffer.deinit();

    // Add experiences
    const state = CubeState.initSolved();
    var next_state = state.clone();
    next_state.step(.U);

    for (0..5) |i| {
        buffer.push(.{
            .state = state,
            .action = @intCast(i),
            .reward = 1.0,
            .next_state = next_state,
            .done = false,
            .episode_id = i,
        });
    }

    try std.testing.expectEqual(@as(usize, 5), buffer.len());
    try std.testing.expect(buffer.canSample(3));

    // Sample batch
    var states: [3 * 324]f32 = undefined;
    var actions: [3]u8 = undefined;
    var rewards: [3]f32 = undefined;
    var next_states: [3 * 324]f32 = undefined;
    var dones: [3]bool = undefined;
    var n_steps_taken: [3]u8 = undefined;
    var indices: [3]usize = undefined;
    var weights: [3]f32 = undefined;

    var prng = std.Random.DefaultPrng.init(42);
    try buffer.sample(
        3,
        0.4, // beta
        prng.random(),
        &states,
        &actions,
        &rewards,
        &next_states,
        &dones,
        &n_steps_taken,
        &indices,
        &weights,
        324,
    );

    // Verify we got data
    for (actions) |action| {
        try std.testing.expect(action < 5);
    }

    // Verify importance weights are normalized (max = 1.0)
    var max_weight: f32 = 0.0;
    for (weights) |w| {
        max_weight = @max(max_weight, w);
        try std.testing.expect(w > 0.0);
        try std.testing.expect(w <= 1.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), max_weight, 1e-5);
}

test "prioritized replay update priorities" {
    const env_mod = @import("../env.zig");
    const CubeState = env_mod.CubeState;

    var buffer = try PrioritizedReplayBuffer(CubeState).init(
        std.testing.allocator,
        10,
        0.6,
        1e-6,
    );
    defer buffer.deinit();

    const state = CubeState.initSolved();

    // Add experiences
    for (0..5) |i| {
        buffer.push(.{
            .state = state,
            .action = @intCast(i),
            .reward = 1.0,
            .next_state = state,
            .done = false,
        });
    }

    // Update priorities based on TD errors
    const indices = [_]usize{ 0, 2, 4 };
    const td_errors = [_]f32{ 10.0, 5.0, 1.0 }; // Different errors

    buffer.updatePriorities(&indices, &td_errors);

    // Check that priorities were updated
    const stats = buffer.getPriorityStats();
    try std.testing.expect(stats.max > stats.min);
    try std.testing.expect(stats.total > 0.0);
}
