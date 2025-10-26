const std = @import("std");

/// Simple 4x4 gridworld for DQN sanity check
/// State: agent position (one-hot 16 dims)
/// Actions: 0=Up, 1=Right, 2=Down, 3=Left
/// Goal: reach (3,3) for +1 reward
/// Pits: (1,1) and (2,2) give -1 reward and reset
pub const GridWorld = struct {
    x: usize,
    y: usize,

    const GRID_SIZE = 4;
    const STATE_DIM = 16; // 4x4 grid one-hot
    const NUM_ACTIONS = 4;

    const GOAL_X = 3;
    const GOAL_Y = 3;

    pub fn init() GridWorld {
        return GridWorld{
            .x = 0,
            .y = 0,
        };
    }

    pub fn reset(self: *GridWorld) void {
        self.x = 0;
        self.y = 0;
    }

    pub fn step(self: *GridWorld, action: u8) struct { reward: f32, done: bool } {
        // Compute distance to goal before move
        const old_dist = manhattanDistance(self.x, self.y, GOAL_X, GOAL_Y);

        // Move based on action
        switch (action) {
            0 => { // Up
                if (self.y > 0) self.y -= 1;
            },
            1 => { // Right
                if (self.x < GRID_SIZE - 1) self.x += 1;
            },
            2 => { // Down
                if (self.y < GRID_SIZE - 1) self.y += 1;
            },
            3 => { // Left
                if (self.x > 0) self.x -= 1;
            },
            else => {},
        }

        // Check for goal
        if (self.x == GOAL_X and self.y == GOAL_Y) {
            return .{ .reward = 1.0, .done = true };
        }

        // Check for pits
        if ((self.x == 1 and self.y == 1) or (self.x == 2 and self.y == 2)) {
            self.reset();
            return .{ .reward = -1.0, .done = true };
        }

        // Compute distance to goal after move
        const new_dist = manhattanDistance(self.x, self.y, GOAL_X, GOAL_Y);

        // Reward shaping: small bonus for getting closer, penalty for getting farther
        const distance_reward = if (new_dist < old_dist)
            @as(f32, 0.1) // Moving closer
        else if (new_dist > old_dist)
            @as(f32, -0.1) // Moving farther
        else
            @as(f32, 0.0); // Same distance (e.g., hit wall)

        // Reduced step penalty + distance-based reward
        const step_penalty: f32 = -0.01;
        return .{ .reward = step_penalty + distance_reward, .done = false };
    }

    fn manhattanDistance(x1: usize, y1: usize, x2: usize, y2: usize) usize {
        const dx = if (x1 > x2) x1 - x2 else x2 - x1;
        const dy = if (y1 > y2) y1 - y2 else y2 - y1;
        return dx + dy;
    }

    pub fn toOneHot(self: GridWorld, out: []f32) void {
        @memset(out, 0.0);
        const idx = self.y * GRID_SIZE + self.x;
        out[idx] = 1.0;
    }

    pub fn hash(self: GridWorld) u64 {
        return @as(u64, self.y) * GRID_SIZE + self.x;
    }
};
