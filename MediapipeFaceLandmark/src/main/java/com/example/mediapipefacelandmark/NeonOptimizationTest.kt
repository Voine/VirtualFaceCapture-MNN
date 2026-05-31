package com.example.mediapipefacelandmark

/**
 * NEON Optimization Test utility.
 * Runs correctness tests and benchmarks for NEON-optimized linear algebra functions.
 *
 * Usage:
 *   // In your Application or Activity onCreate:
 *   NeonOptimizationTest.runTests()
 *
 * Check Logcat with tag "NeonOptTest" to see results.
 */
object NeonOptimizationTest {

    init {
        try {
            System.loadLibrary("mediapipefacelandmark")
        } catch (e: UnsatisfiedLinkError) {
            android.util.Log.e("NeonOptTest", "Failed to load native library: ${e.message}")
        }
    }

    /**
     * Run all NEON optimization tests and benchmarks.
     * Results will be printed to Logcat with tag "NeonOptTest".
     *
     * @return true if all tests passed, false otherwise
     */
    @JvmStatic
    external fun runTests(): Boolean

    /**
     * Check if NEON is enabled on this device.
     *
     * @return true if NEON SIMD is available and enabled
     */
    @JvmStatic
    external fun isNeonEnabled(): Boolean

    /**
     * Run only benchmark tests (skip correctness tests).
     * Useful for performance profiling.
     */
    @JvmStatic
    external fun runBenchmarksOnly()
}
