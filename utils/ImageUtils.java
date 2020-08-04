import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class ImageUtils {
    public static class ImagePanel extends JPanel {
        int SCALE;
        Image img;

        public ImagePanel() {
            this.SCALE = 1;
        }

        public ImagePanel(int scale, Image img) {
            this.SCALE = scale;
            this.img = img;
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2d = (Graphics2D) g;
            g2d.scale(SCALE, SCALE);
            g2d.drawImage((java.awt.Image) this.img.getWrappedImage(), 0, 0, this);
        }
    }

    public static class Container extends JPanel {
        public Container(String label) {
            setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            JLabel l = new JLabel(label, JLabel.CENTER);
            l.setAlignmentX(Component.CENTER_ALIGNMENT);
            add(l);
        }

        public Container(String trueLabel, String predLabel) {
            setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            JLabel l = new JLabel(trueLabel, JLabel.CENTER);
            l.setAlignmentX(Component.CENTER_ALIGNMENT);
            add(l);
            JLabel l2 = new JLabel(predLabel, JLabel.CENTER);
            l2.setAlignmentX(Component.CENTER_ALIGNMENT);
            add(l2);
        }
    }

    public static void showImage(Image img) {
        showImage(img, "", 1);
    }

    public static void showImage(Image img, String name, int SCALE) {
        int WIDTH = img.getWidth();
        int HEIGHT = img.getHeight();
        JFrame frame = new JFrame(name);
        JPanel panel = new ImagePanel(SCALE, img);
        panel.setPreferredSize(new Dimension(WIDTH * SCALE, HEIGHT * SCALE));
        frame.getContentPane().add(panel);
        frame.getContentPane().setLayout(new FlowLayout());
        frame.pack();
        frame.setVisible(true);
    }

    public static void drawBBoxes(Image img, NDArray boxes, String[] labels) {
        System.out.println(boxes.getShape());
        List<String> classNames = new ArrayList();
        List<Double> prob = new ArrayList();
        List<BoundingBox> boundBoxes = new ArrayList();
        for (int i = 0; i < boxes.size(0); i++) {
            NDArray box = boxes.get(i);
            Rectangle rect = bboxToRect(box);
            classNames.add(labels[i]);
            prob.add(1.0);
            boundBoxes.add(rect);
        }
        DetectedObjects detectedObjects = new DetectedObjects(classNames, prob, boundBoxes);
        img.drawBoundingBoxes(detectedObjects);
    }

    public static Rectangle bboxToRect(NDArray bbox) {
        float width = bbox.getFloat(2) - bbox.getFloat(0);
        float height = bbox.getFloat(3) - bbox.getFloat(1);
        return new Rectangle(bbox.getFloat(0), bbox.getFloat(1), width, height);
    }
}
