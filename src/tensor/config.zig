const std = @import("std");

/// Compile-time configuration for tensor operations
pub const TensorConfig = struct {
    /// Element type for tensors (f32 for RL)
    pub const Scalar = f32;

    /// Preferred SIMD vector width (auto-detected or override for testing)
    pub const simd_lanes = std.simd.suggestVectorLength(Scalar) orelse 4;

    /// Preferred memory alignment for SIMD operations
    pub const preferred_alignment = @max(16, simd_lanes * @sizeOf(Scalar));

    /// Enable runtime safety checks in hot paths (disable in release)
    pub const runtime_safety = std.debug.runtime_safety;
};

/// SIMD vector type for tensor operations
pub const Vec = @Vector(TensorConfig.simd_lanes, TensorConfig.Scalar);
