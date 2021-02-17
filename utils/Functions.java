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

    public static Figure plot(double[][] x, double[][] y, String[] traceLabels, String xLabel, String yLabel) {
        ScatterTrace[] traces = new ScatterTrace[x.length];
        for (int i = 0; i < traces.length; i++) {
            traces[i] = ScatterTrace.builder(x[i], y[i])
            .mode(ScatterTrace.Mode.LINE)
            .name(traceLabels[i])
            .build();
        }

        Layout layout = Layout.builder()
                .showLegend(true)
                .xAxis(Axis.builder().title(xLabel).build())
                .yAxis(Axis.builder().title(yLabel).build())
                .build();

        return new Figure(layout, traces);
    }

    public static Figure plotLogScale(double[][] x, double[][] y, String[] traceLabels, String xLabel, String yLabel) {
        ScatterTrace[] traces = new ScatterTrace[x.length];
        for (int i = 0; i < traces.length; i++) {
            traces[i] = ScatterTrace.builder(x[i], y[i])
            .mode(ScatterTrace.Mode.LINE)
            .name(traceLabels[i])
            .build();
        }

        Layout layout = Layout.builder()
                .showLegend(true)
                .xAxis(Axis.builder().type(Axis.Type.LOG).title(xLabel).build())
                .yAxis(Axis.builder().type(Axis.Type.LOG).title(yLabel).build())
                .build();

        return new Figure(layout, traces);
    }
}