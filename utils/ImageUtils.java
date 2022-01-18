import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ImageUtils {

    public static BufferedImage showImages(
            BufferedImage[] images, String[] labels, int width, int height) {
        int col = Math.min(1280 / width, images.length);
        int row = (images.length + col - 1) / col;

        int textHeight = 28;
        int w = col * (width + 3);
        int h = row * (height + 3) + textHeight;
        BufferedImage output = new BufferedImage(w + 3, h + 3, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = output.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setPaint(Color.LIGHT_GRAY);
        g.fill(new java.awt.Rectangle(0, 0, w + 3, h + 3));
        g.setPaint(Color.BLACK);

        Font font = g.getFont();
        FontMetrics metrics = g.getFontMetrics(font);
        for (int i = 0; i < images.length; ++i) {
            int x = (i % col) * (width + 3) + 3;
            int y = (i / col) * (height + 3) + 3;

            int tx = x + (width - metrics.stringWidth(labels[i])) / 2;
            int ty = y + ((textHeight - metrics.getHeight()) / 2) + metrics.getAscent();
            g.drawString(labels[i], tx, ty);

            BufferedImage img = images[i];
            g.drawImage(img, x, y + textHeight, width, height, null);
        }
        g.dispose();
        return output;
    }

    public static BufferedImage showImages(BufferedImage[] images, int width, int height) {
        int col = Math.min(1280 / width, images.length);
        int row = (images.length + col - 1) / col;

        int w = col * (width + 3);
        int h = row * (height + 3);
        BufferedImage output = new BufferedImage(w + 3, h + 3, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = output.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setPaint(Color.LIGHT_GRAY);
        g.fill(new java.awt.Rectangle(0, 0, w + 3, h + 3));
        for (int i = 0; i < images.length; ++i) {
            int x = (i % col) * (width + 3) + 3;
            int y = (i / col) * (height + 3) + 3;

            BufferedImage img = images[i];
            g.drawImage(img, x, y, width, height, null);
        }
        g.dispose();
        return output;
    }

    public static void drawBBoxes(Image img, NDArray boxes, String[] labels) {
        if (labels == null) {
            labels = new String[(int) boxes.size(0)];
            Arrays.fill(labels, "");
        }

        List<String> classNames = new ArrayList<>();
        List<Double> prob = new ArrayList<>();
        List<BoundingBox> boundBoxes = new ArrayList<>();
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
