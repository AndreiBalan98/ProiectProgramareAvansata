package userinterface;

import javax.swing.*;
import java.awt.*;

public class DigitsApplication2 extends JPanel {

    private final double[][][] images;
    private final int imageWidth;
    private final int imageHeight;
    private final int scaleFactor;
    private final int rows;
    private final int cols;

    public DigitsApplication2(double[][][] images, int scaleFactor, int rows, int cols) {
        this.images = images;
        this.imageHeight = images[0].length;
        this.imageWidth = images[0][0].length;
        this.scaleFactor = scaleFactor;
        this.rows = rows;
        this.cols = cols;
        setPreferredSize(new Dimension(cols * imageWidth * scaleFactor, rows * imageHeight * scaleFactor));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        for (int index = 0; index < images.length; index++) {
            int row = index / cols;
            int col = index % cols;

            double[][] data = images[index];
            for (int y = 0; y < imageHeight; y++) {
                for (int x = 0; x < imageWidth; x++) {
                    double value = Math.min(1.0, Math.max(0.0, data[y][x]));
                    int gray = (int) (value * 255);
                    g.setColor(new Color(gray, gray, gray));
                    int drawX = (col * imageWidth + x) * scaleFactor;
                    int drawY = (row * imageHeight + y) * scaleFactor;
                    g.fillRect(drawX, drawY, scaleFactor, scaleFactor);
                }
            }
        }
    }

    public static void renderImage(double[][][] images) {
        int screenWidth = Toolkit.getDefaultToolkit().getScreenSize().width;
        int screenHeight = Toolkit.getDefaultToolkit().getScreenSize().height;

        int numImages = images.length;
        int imageHeight = images[0].length;
        int imageWidth = images[0][0].length;

        // Determină grilă pătratică (aproape de un pătrat)
        int cols = (int) Math.ceil(Math.sqrt(numImages));
        int rows = (int) Math.ceil((double) numImages / cols);

        // Calculează scaleFactor maxim pentru a umple ecranul
        int scaleX = screenWidth / (cols * imageWidth);
        int scaleY = screenHeight / (rows * imageHeight);
        int scaleFactor = Math.max(1, Math.min(scaleX, scaleY)); // cel mai mare posibil dar minim 1

        JFrame frame = new JFrame("Neuron Image Viewer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setExtendedState(JFrame.MAXIMIZED_BOTH); // full screen
        frame.setUndecorated(true); // fără border/title bar

        DigitsApplication2 panel = new DigitsApplication2(images, scaleFactor, rows, cols);
        frame.setContentPane(panel);
        frame.pack();
        frame.setVisible(true);
    }
}

/*
package userinterface;

import javax.swing.*;
import java.awt.*;

public class DigitsApplication2 extends JPanel {

    private final double[][] data;
    private final int scaleFactor;

    public DigitsApplication2(double[][] data, int scaleFactor) {
        this.data = data;
        this.scaleFactor = scaleFactor;
        setPreferredSize(new Dimension(data[0].length * scaleFactor, data.length * scaleFactor));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        for (int y = 0; y < data.length; y++) {
            for (int x = 0; x < data[y].length; x++) {
                double value = Math.min(1.0, Math.max(0.0, data[y][x])); // clamp la [0, 1]
                int gray = (int) (value * 255);
                g.setColor(new Color(gray, gray, gray));
                g.fillRect(x * scaleFactor, y * scaleFactor, scaleFactor, scaleFactor);
            }
        }
    }

    public static void renderImage(double[][] data, int scaleFactor) {
        JFrame frame = new JFrame("Neuron Image Viewer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);

        DigitsApplication2 panel = new DigitsApplication2(data, scaleFactor);
        frame.setContentPane(panel);
        frame.pack();
        frame.setLocationRelativeTo(null); // center on screen
        frame.setVisible(true);
    }
}
*/
