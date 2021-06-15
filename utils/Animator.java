import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * Animates a graph with real-time data.
 *
 * <p>Defined in Ch 3.6 Softmax Reg. from Scratch
 */
public class Animator {

    private String id; // Id reference of graph(for updating graph)
    private Table data; // Data Points

    public Animator() {
        id = "";

        // Incrementally plot data
        data =
                Table.create("Data")
                        .addColumns(
                                FloatColumn.create("epoch", new float[] {}),
                                FloatColumn.create("value", new float[] {}),
                                StringColumn.create("metric", new String[] {}));
    }

    // Add a single metric to the table
    public void add(float epoch, float value, String metric) {
        Row newRow = data.appendRow();
        newRow.setFloat("epoch", epoch);
        newRow.setFloat("value", value);
        newRow.setString("metric", metric);
    }

    // Add accuracy, train accuracy, and train loss metrics for a given epoch
    // Then plot it on the graph
    public void add(float epoch, float accuracy, float trainAcc, float trainLoss) {
        add(epoch, trainLoss, "train loss");
        add(epoch, trainAcc, "train accuracy");
        add(epoch, accuracy, "test accuracy");
        show();
    }

    // Display the graph
    public void show() {
        if (id.equals("")) {
            id = display(LinePlot.create("", data, "epoch", "value", "metric"));
            return;
        }
        update();
    }

    // Update the graph
    public void update() {
        updateDisplay(id, LinePlot.create("", data, "epoch", "value", "metric"));
    }
}
