//TODO: I MUST UNDERSTAND THIS CLASS, ITS FULLY AI GENERATED

package userinterface;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

import neuralnetwork.ForwardNeuralNetwork;

import java.util.stream.IntStream;

public class Application extends JFrame {

    private final ForwardNeuralNetwork network;
    private final BufferedImage canvas;
    private JLabel predictionLabel = null;
    private boolean drawing = false;

    public Application(ForwardNeuralNetwork network) {
        this.network = network;
        this.canvas = new BufferedImage(280, 280, BufferedImage.TYPE_BYTE_GRAY);

        setTitle("Digit Recognizer");
        setSize(320, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        JPanel drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(canvas, 0, 0, getWidth(), getHeight(), null);
            }
        };
        drawPanel.setPreferredSize(new Dimension(280, 280));
        drawPanel.setBackground(Color.WHITE);
        drawPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                drawing = true;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                drawing = false;
                predictDigit();
            }
        });
        drawPanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (drawing) {
                    Graphics2D g2d = canvas.createGraphics();
                    g2d.setColor(Color.BLACK);
                    g2d.fillOval(e.getX(), e.getY(), 20, 20);
                    g2d.dispose();
                    drawPanel.repaint();
                }
            }
        });

        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> {
            Graphics2D g2d = canvas.createGraphics();
            g2d.setPaint(Color.WHITE);
            g2d.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
            g2d.dispose();
            drawPanel.repaint();
            predictionLabel.setText("Prediction: ");
        });

        predictionLabel = new JLabel("Prediction: ", SwingConstants.CENTER);
        predictionLabel.setFont(new Font("Arial", Font.BOLD, 20));

        add(drawPanel, BorderLayout.CENTER);
        add(clearButton, BorderLayout.SOUTH);
        add(predictionLabel, BorderLayout.NORTH);

        setVisible(true);
    }

    private void predictDigit() {
        double[] image = new double[28 * 28];
        for (int y = 0; y < 280; y += 10) {
            for (int x = 0; x < 280; x += 10) {
                int color = new Color(canvas.getRGB(x, y)).getRed();
                image[(y / 10) * 28 + (x / 10)] = 1 - (color / 255.0);
            }
        }

        double[] output = network.feedForward(image);
        int prediction = IntStream.range(0, output.length)
                .reduce((i, j) -> output[i] > output[j] ? i : j)
                .orElse(-1);
        predictionLabel.setText("Prediction: " + prediction);
    }
}