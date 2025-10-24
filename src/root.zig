//! Machine Learning library in Zig with SIMD acceleration
//!
//! Designed for reinforcement learning on Rubik's Cube
const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const nn = @import("nn.zig");
pub const env = @import("env.zig");
pub const rl = @import("rl.zig");

test {
    std.testing.refAllDecls(@This());
}
