const std = @import("std");

/// Experience tuple for replay buffer (generic over state type)
pub fn Experience(comptime StateType: type) type {
    return struct {
        state: StateType,
        action: u8,
        reward: f32,
        next_state: StateType,
        done: bool,
        episode_id: usize = 0, // For tracking buffer age diagnostics
    };
}

/// Ring buffer for experience replay (generic over state type)
pub fn ReplayBuffer(comptime StateType: type) type {
    const Exp = Experience(StateType);

    return struct {
        buffer: []Exp,
        capacity: usize,
        size: usize,
        pos: usize,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            const buffer = try allocator.alloc(Exp, capacity);
            return Self{
                .buffer = buffer,
                .capacity = capacity,
                .size = 0,
                .pos = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }

        /// Add an experience to the buffer
        pub fn push(self: *Self, exp: Exp) void {
            self.buffer[self.pos] = exp;
            self.pos = (self.pos + 1) % self.capacity;
            if (self.size < self.capacity) {
                self.size += 1;
            }
        }

        /// Sample a batch of experiences
        /// Requires state_dim parameter to know how large the one-hot encoding is
        pub fn sample(
            self: *Self,
            batch_size: usize,
            rng: std.Random,
            states: []f32,
            actions: []u8,
            rewards: []f32,
            next_states: []f32,
            dones: []bool,
            state_dim: usize,
        ) !void {
            if (self.size < batch_size) {
                return error.InsufficientSamples;
            }

            // Sample random indices (with replacement for simplicity)
            for (0..batch_size) |i| {
                const idx = rng.intRangeLessThan(usize, 0, self.size);
                const exp = &self.buffer[idx];

                // Convert state to one-hot
                const state_offset = i * state_dim;
                exp.state.toOneHot(states[state_offset .. state_offset + state_dim]);

                // Convert next_state to one-hot
                const next_state_offset = i * state_dim;
                exp.next_state.toOneHot(next_states[next_state_offset .. next_state_offset + state_dim]);

                actions[i] = exp.action;
                rewards[i] = exp.reward;
                dones[i] = exp.done;
            }
        }

        /// Check if buffer has enough samples to train
        pub fn canSample(self: *Self, batch_size: usize) bool {
            return self.size >= batch_size;
        }

        /// Get current buffer size
        pub fn len(self: *Self) usize {
            return self.size;
        }

        /// Get buffer age diagnostics (min/max episode ID)
        pub fn getAgeStats(self: *Self) struct { min_episode: usize, max_episode: usize } {
            if (self.size == 0) {
                return .{ .min_episode = 0, .max_episode = 0 };
            }

            var min_ep = self.buffer[0].episode_id;
            var max_ep = self.buffer[0].episode_id;

            for (self.buffer[0..self.size]) |exp| {
                min_ep = @min(min_ep, exp.episode_id);
                max_ep = @max(max_ep, exp.episode_id);
            }

            return .{ .min_episode = min_ep, .max_episode = max_ep };
        }
    };
}

// Tests using 3x3 cube
test "replay buffer push and sample" {
    const env_mod = @import("../env.zig");
    const CubeState = env_mod.CubeState;

    var buffer = try ReplayBuffer(CubeState).init(std.testing.allocator, 10);
    defer buffer.deinit();

    // Add some experiences
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
        });
    }

    try std.testing.expectEqual(@as(usize, 5), buffer.len());
    try std.testing.expect(buffer.canSample(3));

    // Sample a batch
    var states: [3 * 324]f32 = undefined;
    var actions: [3]u8 = undefined;
    var rewards: [3]f32 = undefined;
    var next_states: [3 * 324]f32 = undefined;
    var dones: [3]bool = undefined;

    var prng = std.Random.DefaultPrng.init(42);
    try buffer.sample(3, prng.random(), &states, &actions, &rewards, &next_states, &dones, 324);

    // Verify we got data
    for (actions) |action| {
        try std.testing.expect(action < 5);
    }

    for (rewards) |reward| {
        try std.testing.expectEqual(@as(f32, 1.0), reward);
    }
}

test "replay buffer ring behavior" {
    const env_mod = @import("../env.zig");
    const CubeState = env_mod.CubeState;

    var buffer = try ReplayBuffer(CubeState).init(std.testing.allocator, 3);
    defer buffer.deinit();

    const state = CubeState.initSolved();

    // Fill buffer
    for (0..3) |i| {
        buffer.push(.{
            .state = state,
            .action = @intCast(i),
            .reward = 0.0,
            .next_state = state,
            .done = false,
        });
    }

    try std.testing.expectEqual(@as(usize, 3), buffer.len());

    // Add one more - should wrap around
    buffer.push(.{
        .state = state,
        .action = 99,
        .reward = 0.0,
        .next_state = state,
        .done = false,
    });

    // Size should still be 3 (capacity)
    try std.testing.expectEqual(@as(usize, 3), buffer.len());
    try std.testing.expectEqual(@as(usize, 1), buffer.pos);
}
