public class PlotUtils {
    /* Helper function to plot by a single or multiple lines */
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

    /* Helper function to plot by a single or multiple lines with Log scale */
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