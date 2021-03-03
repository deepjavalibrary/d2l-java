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

    /**
     * Return the i'th GPU if it exists, otherwise return the CPU 
     */
    public static Device tryGpu(int i) {
        return Device.getGpuCount() >= i + 1 ? Device.gpu(i) : Device.cpu();
    }

    @FunctionalInterface
    public interface TriFunction<T, U, V, W> {
        public W apply(T t, U u, V v);
    }

    @FunctionalInterface
    public interface QuadFunction<T, U, V, W, R> {
        public R apply(T t, U u, V v, W w);
    }

    @FunctionalInterface
    public interface SimpleFunction<T> {
        public T apply();
    }

    @FunctionalInterface
    public interface voidFunction<T> {
        public void apply(T t);
    }

    @FunctionalInterface
    public interface voidTwoFunction<T, U> {
        public void apply(T t, U u);
    }
}