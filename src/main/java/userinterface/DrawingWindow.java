package userinterface;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import neuralnetwork.DigitsNN;

public class DrawingWindow {

    private static final int CANVAS_SIZE = 280; // 28x28 scaled up by 10
    private static final int GRID_SIZE = 28;
    private static final double CELL_SIZE = (double) CANVAS_SIZE / GRID_SIZE;

    private DigitsNN model;
    private Canvas canvas;
    private GraphicsContext gc;
    private double[][] pixelData;
    private Label[] confidenceLabels;
    private ProgressBar[] confidenceBars;
    private Stage stage;

    public DrawingWindow(DigitsNN model) {
        this.model = model;
        this.pixelData = new double[GRID_SIZE][GRID_SIZE];
        initializeWindow();
    }

    private void initializeWindow() {
        stage = new Stage();
        stage.setTitle("Draw a Digit");

        // Main layout
        HBox mainLayout = new HBox(20);
        mainLayout.setPadding(new Insets(20));
        mainLayout.setStyle("-fx-background-color: #f5f5f5;");

        // Left side - Drawing area
        VBox leftSide = new VBox(15);
        leftSide.setAlignment(Pos.CENTER);

        Label instructionLabel = new Label("Draw a digit (0-9):");
        instructionLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

        canvas = new Canvas(CANVAS_SIZE, CANVAS_SIZE);
        canvas.setStyle("-fx-border-color: #333; -fx-border-width: 2px;");
        gc = canvas.getGraphicsContext2D();

        // Initialize canvas
        clearCanvas();

        // Mouse events for drawing
        canvas.setOnMousePressed(this::handleMousePressed);
        canvas.setOnMouseDragged(this::handleMouseDragged);
        canvas.setOnMouseReleased(this::handleMouseReleased);

        Button clearBtn = new Button("Clear");
        clearBtn.setStyle("-fx-background-color: #f44336; -fx-text-fill: white; -fx-font-size: 14px;");
        clearBtn.setPrefSize(100, 35);
        clearBtn.setOnAction(e -> {
            clearCanvas();
            clearPredictions();
        });

        Button predictBtn = new Button("Predict");
        predictBtn.setStyle("-fx-background-color: #4CAF50; -fx-text-fill: white; -fx-font-size: 14px;");
        predictBtn.setPrefSize(100, 35);
        predictBtn.setOnAction(e -> makePrediction());

        HBox buttonBox = new HBox(10);
        buttonBox.setAlignment(Pos.CENTER);
        buttonBox.getChildren().addAll(clearBtn, predictBtn);

        leftSide.getChildren().addAll(instructionLabel, canvas, buttonBox);

        // Right side - Predictions
        VBox rightSide = new VBox(10);
        rightSide.setAlignment(Pos.TOP_LEFT);
        rightSide.setPrefWidth(250);

        Label predictionLabel = new Label("Confidence Rates:");
        predictionLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

        // Create confidence display for each digit
        confidenceLabels = new Label[10];
        confidenceBars = new ProgressBar[10];

        VBox confidenceBox = new VBox(8);
        for (int i = 0; i < 10; i++) {
            HBox digitBox = new HBox(10);
            digitBox.setAlignment(Pos.CENTER_LEFT);

            Label digitLabel = new Label("Digit " + i + ":");
            digitLabel.setPrefWidth(60);
            digitLabel.setStyle("-fx-font-weight: bold;");

            confidenceBars[i] = new ProgressBar(0);
            confidenceBars[i].setPrefWidth(120);
            confidenceBars[i].setStyle("-fx-accent: #2196F3;");

            confidenceLabels[i] = new Label("0.00%");
            confidenceLabels[i].setPrefWidth(60);
            confidenceLabels[i].setStyle("-fx-font-family: monospace;");

            digitBox.getChildren().addAll(digitLabel, confidenceBars[i], confidenceLabels[i]);
            confidenceBox.getChildren().add(digitBox);
        }

        rightSide.getChildren().addAll(predictionLabel, confidenceBox);

        mainLayout.getChildren().addAll(leftSide, rightSide);

        Scene scene = new Scene(mainLayout, 650, 400);
        stage.setScene(scene);
    }

    private void clearCanvas() {
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

        // Draw grid lines (optional, for better visualization)
        gc.setStroke(Color.LIGHTGRAY);
        gc.setLineWidth(0.5);
        for (int i = 0; i <= GRID_SIZE; i++) {
            double pos = i * CELL_SIZE;
            gc.strokeLine(pos, 0, pos, CANVAS_SIZE);
            gc.strokeLine(0, pos, CANVAS_SIZE, pos);
        }

        // Clear pixel data
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                pixelData[i][j] = 0.0;
            }
        }
    }

    private void clearPredictions() {
        for (int i = 0; i < 10; i++) {
            confidenceLabels[i].setText("0.00%");
            confidenceBars[i].setProgress(0);
        }
    }

    private void handleMousePressed(MouseEvent e) {
        drawAt(e.getX(), e.getY());
    }

    private void handleMouseDragged(MouseEvent e) {
        drawAt(e.getX(), e.getY());
    }

    private void handleMouseReleased(MouseEvent e) {
        // Auto-predict after drawing
        makePrediction();
    }

    private void drawAt(double x, double y) {
        if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) return;

        int gridX = (int) (x / CELL_SIZE);
        int gridY = (int) (y / CELL_SIZE);

        // Draw on canvas
        gc.setFill(Color.BLACK);
        double cellX = gridX * CELL_SIZE;
        double cellY = gridY * CELL_SIZE;
        gc.fillRect(cellX, cellY, CELL_SIZE, CELL_SIZE);

        // Update pixel data
        pixelData[gridY][gridX] = 1.0;

        // Also draw on neighboring cells for better drawing experience
        drawNeighbors(gridX, gridY);
    }

    private void drawNeighbors(int centerX, int centerY) {
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int x = centerX + dx;
                int y = centerY + dy;
                if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
                    double intensity = (dx == 0 && dy == 0) ? 1.0 : 0.3;
                    if (pixelData[y][x] < intensity) {
                        pixelData[y][x] = intensity;

                        // Update canvas display
                        double alpha = intensity;
                        gc.setFill(Color.color(0, 0, 0, alpha));
                        gc.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                    }
                }
            }
        }
    }

    private void makePrediction() {
        // Convert 2D pixel data to 1D array for neural network
        double[] input = new double[GRID_SIZE * GRID_SIZE];
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                input[i * GRID_SIZE + j] = pixelData[i][j];
            }
        }

        // Get prediction from neural network
        double[] output = model.feedForward(input);

        // Apply softmax to get probabilities
        double[] probabilities = softmax(output);

        // Update UI with predictions
        for (int i = 0; i < 10; i++) {
            double confidence = probabilities[i] * 100;
            confidenceLabels[i].setText(String.format("%.2f%%", confidence));
            confidenceBars[i].setProgress(probabilities[i]);

            // Highlight the most confident prediction
            if (probabilities[i] == getMaxValue(probabilities)) {
                confidenceLabels[i].setStyle("-fx-font-family: monospace; -fx-font-weight: bold; -fx-text-fill: #4CAF50;");
                confidenceBars[i].setStyle("-fx-accent: #4CAF50;");
            } else {
                confidenceLabels[i].setStyle("-fx-font-family: monospace; -fx-font-weight: normal; -fx-text-fill: black;");
                confidenceBars[i].setStyle("-fx-accent: #2196F3;");
            }
        }
    }

    private double[] softmax(double[] input) {
        double[] result = new double[input.length];
        double max = getMaxValue(input);
        double sum = 0;

        // Subtract max for numerical stability and compute exponentials
        for (int i = 0; i < input.length; i++) {
            result[i] = Math.exp(input[i] - max);
            sum += result[i];
        }

        // Normalize
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    private double getMaxValue(double[] array) {
        double max = array[0];
        for (double value : array) {
            if (value > max) max = value;
        }
        return max;
    }

    public void show() {
        stage.show();
    }
}