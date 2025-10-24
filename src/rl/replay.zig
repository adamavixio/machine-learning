const std = @import("std");
const env_mod = @import("../env.zig");

const CubeState = env_mod.CubeState;

/// Experience tuple for replay buffer
pub const Experience = struct {
    state: CubeState,
    action: u8, // Move index (0-11)
    reward: f32,
    next_state: CubeState,
    done: bool,
};

/// Ring buffer for experience replay
pub const ReplayBuffer = struct {
    buffer: []Experience,
    capacity: usize,
    size: usize,
    pos: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !ReplayBuffer {
        const buffer = try allocator.alloc(Experience, capacity);
        return ReplayBuffer{
            .buffer = buffer,
            .capacity = capacity,
            .size = 0,
            .pos = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ReplayBuffer) void {
        self.allocator.free(self.buffer);
    }

    /// Add an experience to the buffer
    pub fn push(self: *ReplayBuffer, exp: Experience) void {
        self.buffer[self.pos] = exp;
        self.pos = (self.pos + 1) % self.capacity;
        if (self.size < self.capacity) {
            self.size += 1;
        }
    }

    /// Sample a batch of experiences
    /// Returns contiguous tensors: states [batch, 324], actions [batch], rewards [batch],
    /// next_states [batch, 324], dones [batch]
    pub fn sample(
        self: *ReplayBuffer,
        batch_size: usize,
        rng: std.Random,
        states: []f32,
        actions: []u8,
        rewards: []f32,
        next_states: []f32,
        dones: []bool,
    ) !void {
        if (self.size < batch_size) {
            return error.InsufficientSamples;
        }

        // Sample random indices without replacement (simple version - allow replacement)
        for (0..batch_size) |i| {
            const idx = rng.intRangeLessThan(usize, 0, self.size);
            const exp = &self.buffer[idx];

            // Convert state to one-hot
            const state_offset = i * 324;
            exp.state.toOneHot(states[state_offset .. state_offset + 324]);

            // Convert next_state to one-hot
            const next_state_offset = i * 324;
            exp.next_state.toOneHot(next_states[next_state_offset .. next_state_offset + 324]);

            actions[i] = exp.action;
            rewards[i] = exp.reward;
            dones[i] = exp.done;
        }
    }

    /// Check if buffer has enough samples to train
    pub fn canSample(self: *ReplayBuffer, batch_size: usize) bool {
        return self.size >= batch_size;
    }

    /// Get current buffer size
    pub fn len(self: *ReplayBuffer) usize {
        return self.size;
    }
};

test "replay buffer push and sample" {
    var buffer = try ReplayBuffer.init(std.testing.allocator, 10);
    defer buffer.deinit();

    // Add some experiences
    const state = CubeState.initSolved();
    var next_state = state.clone();
    next_state.step(.U);

    for (0..5) |i| {
        buffer.push(Experience{
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
    try buffer.sample(3, prng.random(), &states, &actions, &rewards, &next_states, &dones);

    // Verify we got data
    for (actions) |action| {
        try std.testing.expect(action < 5);
    }

    for (rewards) |reward| {
        try std.testing.expectEqual(@as(f32, 1.0), reward);
    }
}

test "replay buffer ring behavior" {
    var buffer = try ReplayBuffer.init(std.testing.allocator, 3);
    defer buffer.deinit();

    const state = CubeState.initSolved();

    // Fill buffer
    for (0..3) |i| {
        buffer.push(Experience{
            .state = state,
            .action = @intCast(i),
            .reward = 0.0,
            .next_state = state,
            .done = false,
        });
    }

    try std.testing.expectEqual(@as(usize, 3), buffer.len());

    // Add one more - should wrap around
    buffer.push(Experience{
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
