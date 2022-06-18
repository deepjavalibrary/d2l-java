import ai.djl.ndarray.NDArray;

import tech.tablesaw.plotly.components.*;
import tech.tablesaw.plotly.traces.*;

public class PlotUtils {
    /* Helper function to plot by a single or multiple lines */
    public static Figure plot(
            double[][] x, double[][] y, String[] traceLabels, String xLabel, String yLabel) {
        ScatterTrace[] traces = new ScatterTrace[x.length];
        for (int i = 0; i < traces.length; i++) {
            traces[i] =
                    ScatterTrace.builder(x[i], y[i])
                            .mode(ScatterTrace.Mode.LINE)
                            .name(traceLabels[i])
                            .build();
        }

        Layout layout =
                Layout.builder()
                        .showLegend(true)
                        .xAxis(Axis.builder().title(xLabel).build())
                        .yAxis(Axis.builder().title(yLabel).build())
                        .build();

        return new Figure(layout, traces);
    }

    /* Helper function to plot by a single or multiple lines with Log scale */
    public static Figure plotLogScale(
            double[][] x, double[][] y, String[] traceLabels, String xLabel, String yLabel) {
        ScatterTrace[] traces = new ScatterTrace[x.length];
        for (int i = 0; i < traces.length; i++) {
            traces[i] =
                    ScatterTrace.builder(x[i], y[i])
                            .mode(ScatterTrace.Mode.LINE)
                            .name(traceLabels[i])
                            .build();
        }

        Layout layout =
                Layout.builder()
                        .showLegend(true)
                        .xAxis(Axis.builder().type(Axis.Type.LOG).title(xLabel).build())
                        .yAxis(Axis.builder().type(Axis.Type.LOG).title(yLabel).build())
                        .build();

        return new Figure(layout, traces);
    }

    public static Figure showHeatmaps(
            NDArray matrices,
            String xLabel,
            String yLabel,
            String[] titles,
            int width,
            int height) {
        int numRows = (int) matrices.getShape().get(0);
        int numCols = (int) matrices.getShape().get(1);

        Trace[] traces = new Trace[numRows * numCols];
        int count = 0;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                NDArray NDMatrix = matrices.get(i).get(j);
                double[][] matrix =
                        new double[(int) NDMatrix.getShape().get(0)]
                                [(int) NDMatrix.getShape().get(1)];
                Object[] x = new Object[matrix.length];
                Object[] y = new Object[matrix.length];
                for (int k = 0; k < NDMatrix.getShape().get(0); k++) {
                    matrix[k] = Functions.floatToDoubleArray(NDMatrix.get(k).toFloatArray());
                    x[k] = k;
                    y[k] = k;
                }
                HeatmapTrace.HeatmapBuilder builder = HeatmapTrace.builder(x, y, matrix);
                if (titles != null) {
                    builder = (HeatmapTrace.HeatmapBuilder) builder.name(titles[j]);
                }
                traces[count++] = builder.build();
            }
        }
        Grid grid =
                Grid.builder()
                        .columns(numCols)
                        .rows(numRows)
                        .pattern(Grid.Pattern.INDEPENDENT)
                        .build();
        Layout layout =
                Layout.builder()
                        .title("")
                        .xAxis(Axis.builder().title(xLabel).build())
                        .yAxis(Axis.builder().title(yLabel).build())
                        .width(width)
                        .height(height)
                        .grid(grid)
                        .build();
        return new Figure(layout, traces);
    }
}
