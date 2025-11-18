/// Optimal solvers for Rubik's Cube environments
const std = @import("std");

pub const bfs = @import("solver/bfs.zig");

test {
    std.testing.refAllDecls(@This());
}
