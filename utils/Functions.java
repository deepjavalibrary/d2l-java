import ai.djl.Device;
import ai.djl.engine.Engine;

import java.util.function.Function;

public class Functions {
    // Applies the function `func` to `x` element-wise
    // Returns a new float array with the result
    public static float[] callFunc(float[] x, Function<Float, Float> func) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = func.apply(x[i]);
        }
        return y;
    }

    // ScatterTrace.builder() does not support float[],
    // so we must convert to a double array first
    public static double[] floatToDoubleArray(float[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = x[i];
        }
        return ret;
    }

    /** Return the i'th GPU if it exists, otherwise return the CPU */
    public static Device tryGpu(int i) {
        return Engine.getInstance().getGpuCount() > i ? Device.gpu(i) : Device.cpu();
    }

    /**
     * Helper function to later be able to use lambda. Accepts three types for parameters and one
     * for output.
     */
    @FunctionalInterface
    public interface TriFunction<T, U, V, W> {
        W apply(T t, U u, V v);
    }

    /**
     * Helper function to later be able to use lambda. Accepts 4 types for parameters and one for
     * output.
     */
    @FunctionalInterface
    public interface QuadFunction<T, U, V, W, R> {
        R apply(T t, U u, V v, W w);
    }

    /**
     * Helper function to later be able to use lambda. Doesn't have any type for parameters and has
     * one type for output.
     */
    @FunctionalInterface
    public interface SimpleFunction<T> {
        T apply();
    }

    /**
     * Helper function to later be able to use lambda. Accepts one types for parameters and uses
     * void for return.
     */
    @FunctionalInterface
    public interface voidFunction<T> {
        void apply(T t);
    }

    /**
     * Helper function to later be able to use lambda. Accepts two types for parameters and uses
     * void for return.
     */
    @FunctionalInterface
    public interface voidTwoFunction<T, U> {
        void apply(T t, U u);
    }
}
