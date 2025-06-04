package userinterface;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import neuralnetwork.DigitsNN;
import utils.ModelSaver;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

public class DigitsApplication {

    private Stage primaryStage;
    private DigitsNN currentModel;

    public void start(Stage primaryStage) {
        this.primaryStage = primaryStage;
        primaryStage.setTitle("MNIST Digits Dataset");
        primaryStage.setResizable(false);

        // Create main menu
        VBox mainLayout = new VBox(20);
        mainLayout.setAlignment(Pos.CENTER);
        mainLayout.setPadding(new Insets(50));
        mainLayout.setStyle("-fx-background-color: #f0f0f0;");

        Label titleLabel = new Label("MNIST Digits Neural Network");
        titleLabel.setStyle("-fx-font-size: 24px; -fx-font-weight: bold; -fx-text-fill: #333;");

        Button createModelBtn = new Button("Create Model");
        createModelBtn.setPrefSize(200, 50);
        createModelBtn.setStyle("-fx-font-size: 16px; -fx-background-color: #4CAF50; -fx-text-fill: white;");
        createModelBtn.setOnAction(e -> showCreateModelWindow());

        Button testModelBtn = new Button("Test Model");
        testModelBtn.setPrefSize(200, 50);
        testModelBtn.setStyle("-fx-font-size: 16px; -fx-background-color: #2196F3; -fx-text-fill: white;");
        testModelBtn.setOnAction(e -> showTestModelWindow());

        mainLayout.getChildren().addAll(titleLabel, createModelBtn, testModelBtn);

        Scene scene = new Scene(mainLayout, 400, 300);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void showCreateModelWindow() {
        Stage createStage = new Stage();
        createStage.setTitle("Create Neural Network Model");
        createStage.initOwner(primaryStage);

        VBox layout = new VBox(15);
        layout.setPadding(new Insets(20));
        layout.setStyle("-fx-background-color: #f8f8f8;");

        // Model parameters
        GridPane paramGrid = new GridPane();
        paramGrid.setHgap(10);
        paramGrid.setVgap(10);
        paramGrid.setAlignment(Pos.CENTER);

        TextField modelNameField = new TextField("my_model");
        TextField inputSizeField = new TextField("784");
        TextField hiddenLayersField = new TextField("2");
        TextField hiddenSizesField = new TextField("128,64");
        TextField outputSizeField = new TextField("10");
        CheckBox xavierCheckBox = new CheckBox();
        xavierCheckBox.setSelected(true);

        paramGrid.add(new Label("Model Name:"), 0, 0);
        paramGrid.add(modelNameField, 1, 0);
        paramGrid.add(new Label("Input Size:"), 0, 1);
        paramGrid.add(inputSizeField, 1, 1);
        paramGrid.add(new Label("Hidden Layers:"), 0, 2);
        paramGrid.add(hiddenLayersField, 1, 2);
        paramGrid.add(new Label("Hidden Sizes:"), 0, 3);
        paramGrid.add(hiddenSizesField, 1, 3);
        paramGrid.add(new Label("Output Size:"), 0, 4);
        paramGrid.add(outputSizeField, 1, 4);
        paramGrid.add(new Label("Xavier Init:"), 0, 5);
        paramGrid.add(xavierCheckBox, 1, 5);

        // Training parameters
        Separator separator = new Separator();
        Label trainLabel = new Label("Training Parameters");
        trainLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

        GridPane trainGrid = new GridPane();
        trainGrid.setHgap(10);
        trainGrid.setVgap(10);
        trainGrid.setAlignment(Pos.CENTER);

        TextField epochsField = new TextField("3");
        TextField learningRateField = new TextField("0.01");
        TextField trainSizeField = new TextField("500");
        TextField testSizeField = new TextField("100");
        TextField batchSizeField = new TextField("32");

        trainGrid.add(new Label("Epochs:"), 0, 0);
        trainGrid.add(epochsField, 1, 0);
        trainGrid.add(new Label("Learning Rate:"), 0, 1);
        trainGrid.add(learningRateField, 1, 1);
        trainGrid.add(new Label("Train Size:"), 0, 2);
        trainGrid.add(trainSizeField, 1, 2);
        trainGrid.add(new Label("Test Size:"), 0, 3);
        trainGrid.add(testSizeField, 1, 3);
        trainGrid.add(new Label("Batch Size:"), 0, 4);
        trainGrid.add(batchSizeField, 1, 4);

        // Console area
        TextArea consoleArea = new TextArea();
        consoleArea.setEditable(false);
        consoleArea.setPrefRowCount(12);
        consoleArea.setStyle("-fx-font-family: monospace; -fx-background-color: #000; -fx-text-fill: #0f0;");

        Button startTrainingBtn = new Button("Start Training");
        startTrainingBtn.setStyle("-fx-background-color: #FF9800; -fx-text-fill: white; -fx-font-size: 14px;");
        startTrainingBtn.setPrefSize(150, 40);

        // Add a progress bar
        ProgressBar progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(300);
        progressBar.setVisible(false);

        startTrainingBtn.setOnAction(e -> {
            try {
                // Validate and parse parameters
                String modelName = modelNameField.getText().trim();
                if (modelName.isEmpty()) {
                    showError("Model name cannot be empty");
                    return;
                }

                int inputSize = parseIntField(inputSizeField, "Input Size");
                int hiddenLayers = parseIntField(hiddenLayersField, "Hidden Layers");
                String[] hiddenSizesStr = hiddenSizesField.getText().split(",");
                int[] hiddenSizes = new int[hiddenSizesStr.length];

                for (int i = 0; i < hiddenSizesStr.length; i++) {
                    try {
                        hiddenSizes[i] = Integer.parseInt(hiddenSizesStr[i].trim());
                        if (hiddenSizes[i] <= 0) {
                            showError("Hidden layer sizes must be positive numbers");
                            return;
                        }
                    } catch (NumberFormatException ex) {
                        showError("Invalid hidden layer size: " + hiddenSizesStr[i]);
                        return;
                    }
                }

                if (hiddenSizes.length != hiddenLayers) {
                    showError("Number of hidden layer sizes must match number of hidden layers");
                    return;
                }

                int outputSize = parseIntField(outputSizeField, "Output Size");
                boolean xavier = xavierCheckBox.isSelected();

                int epochs = parseIntField(epochsField, "Epochs");
                double learningRate = parseDoubleField(learningRateField, "Learning Rate");
                int trainSize = parseIntField(trainSizeField, "Train Size");
                int testSize = parseIntField(testSizeField, "Test Size");
                int batchSize = parseIntField(batchSizeField, "Batch Size");

                // Validate ranges
                if (epochs <= 0 || epochs > 100) {
                    showError("Epochs must be between 1 and 100");
                    return;
                }
                if (learningRate <= 0 || learningRate > 1) {
                    showError("Learning rate must be between 0 and 1");
                    return;
                }
                if (trainSize <= 0 || trainSize > 60000) {
                    showError("Train size must be between 1 and 60000");
                    return;
                }
                if (testSize <= 0 || testSize > 10000) {
                    showError("Test size must be between 1 and 10000");
                    return;
                }
                if (batchSize <= 0 || batchSize > trainSize) {
                    showError("Batch size must be between 1 and train size");
                    return;
                }

                // Disable controls during training
                startTrainingBtn.setDisable(true);
                progressBar.setVisible(true);
                consoleArea.clear();
                consoleArea.appendText("=== MNIST Neural Network Training ===\n");
                consoleArea.appendText("Model: " + modelName + "\n");
                consoleArea.appendText("Architecture: " + inputSize + " -> ");
                for (int size : hiddenSizes) {
                    consoleArea.appendText(size + " -> ");
                }
                consoleArea.appendText(outputSize + "\n");
                consoleArea.appendText("Training parameters: " + epochs + " epochs, LR=" + learningRate + "\n");
                consoleArea.appendText("Dataset: " + trainSize + " train, " + testSize + " test\n");
                consoleArea.appendText("=====================================\n\n");

                // Create and train model in background thread
                Task<Void> trainingTask = new Task<Void>() {
                    @Override
                    protected Void call() throws Exception {
                        try {
                            Platform.runLater(() -> consoleArea.appendText("Creating neural network...\n"));

                            currentModel = new DigitsNN(inputSize, hiddenLayers, hiddenSizes, outputSize, !xavier);

                            Platform.runLater(() -> consoleArea.appendText("Neural network created successfully!\n"));

                            // Train the model with console output
                            currentModel.trainWithConsole(epochs, learningRate, trainSize, testSize, batchSize,
                                    (message) -> Platform.runLater(() -> {
                                        consoleArea.appendText(message + "\n");
                                        consoleArea.setScrollTop(Double.MAX_VALUE); // Auto-scroll
                                    }));

                            Platform.runLater(() -> {
                                consoleArea.appendText("\n=== TRAINING COMPLETED ===\n");
                                consoleArea.appendText("Saving model...\n");
                            });

                            // Save model
                            ModelSaver.saveModel(currentModel, modelName);

                            Platform.runLater(() -> {
                                consoleArea.appendText("Model saved successfully as: " + modelName + ".model\n");
                                consoleArea.appendText("You can now test the model using 'Test Model' button.\n");
                            });

                        } catch (Exception ex) {
                            // Log the full stack trace to console
                            Platform.runLater(() -> {
                                consoleArea.appendText("\n=== ERROR OCCURRED ===\n");
                                consoleArea.appendText("Error: " + ex.getClass().getSimpleName() + "\n");
                                consoleArea.appendText("Message: " + (ex.getMessage() != null ? ex.getMessage() : "No message available") + "\n");

                                if (ex.getCause() != null) {
                                    consoleArea.appendText("Caused by: " + ex.getCause().getClass().getSimpleName() + "\n");
                                    consoleArea.appendText("Cause message: " + (ex.getCause().getMessage() != null ? ex.getCause().getMessage() : "No cause message") + "\n");
                                }

                                consoleArea.appendText("\nFull stack trace:\n");
                                StringWriter sw = new StringWriter();
                                PrintWriter pw = new PrintWriter(sw);
                                ex.printStackTrace(pw);
                                consoleArea.appendText(sw.toString());
                                consoleArea.appendText("\n======================\n");
                            });
                            throw ex; // Re-throw for task failure handling
                        }

                        return null;
                    }
                };

                trainingTask.setOnSucceeded(event -> {
                    startTrainingBtn.setDisable(false);
                    progressBar.setVisible(false);
                });

                trainingTask.setOnFailed(event -> {
                    Throwable exception = trainingTask.getException();
                    Platform.runLater(() -> {
                        consoleArea.appendText("\n=== TRAINING FAILED ===\n");
                        if (exception != null) {
                            consoleArea.appendText("Final error: " + exception.getMessage() + "\n");
                        }
                        consoleArea.appendText("Please check the error details above and try again.\n");
                        startTrainingBtn.setDisable(false);
                        progressBar.setVisible(false);
                    });
                });

                new Thread(trainingTask).start();

            } catch (Exception ex) {
                showError("Configuration error: " + ex.getMessage());
            }
        });

        layout.getChildren().addAll(
                new Label("Model Configuration"),
                paramGrid,
                separator,
                trainLabel,
                trainGrid,
                new Label("Training Console:"),
                consoleArea,
                progressBar,
                startTrainingBtn
        );

        ScrollPane scrollPane = new ScrollPane(layout);
        scrollPane.setFitToWidth(true);
        Scene scene = new Scene(scrollPane, 550, 750);
        createStage.setScene(scene);
        createStage.show();
    }

    private int parseIntField(TextField field, String fieldName) throws NumberFormatException {
        try {
            int value = Integer.parseInt(field.getText().trim());
            if (value <= 0) {
                throw new NumberFormatException(fieldName + " must be a positive number");
            }
            return value;
        } catch (NumberFormatException e) {
            throw new NumberFormatException("Invalid " + fieldName + ": " + field.getText());
        }
    }

    private double parseDoubleField(TextField field, String fieldName) throws NumberFormatException {
        try {
            double value = Double.parseDouble(field.getText().trim());
            if (value <= 0) {
                throw new NumberFormatException(fieldName + " must be a positive number");
            }
            return value;
        } catch (NumberFormatException e) {
            throw new NumberFormatException("Invalid " + fieldName + ": " + field.getText());
        }
    }

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText("Configuration Error");
        alert.setContentText(message);
        alert.showAndWait();
    }

    private void showTestModelWindow() {
        // First, let user select a saved model
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Saved Model");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Model Files", "*.model")
        );

        File selectedFile = fileChooser.showOpenDialog(primaryStage);
        if (selectedFile == null) return;

        try {
            DigitsNN loadedModel = ModelSaver.loadModel(selectedFile.getAbsolutePath());

            // Open drawing window
            DrawingWindow drawingWindow = new DrawingWindow(loadedModel);
            drawingWindow.show();

        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Error");
            alert.setHeaderText("Failed to load model");
            alert.setContentText("Error: " + e.getClass().getSimpleName() + "\n" +
                    "Message: " + (e.getMessage() != null ? e.getMessage() : "Unknown error") + "\n\n" +
                    "Please make sure the selected file is a valid model file.");
            alert.showAndWait();
        }
    }
}