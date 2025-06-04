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

        TextField epochsField = new TextField("5");
        TextField learningRateField = new TextField("0.01");
        TextField trainSizeField = new TextField("1000");
        TextField testSizeField = new TextField("100");
        TextField batchSizeField = new TextField("100");

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
        consoleArea.setPrefRowCount(10);
        consoleArea.setStyle("-fx-font-family: monospace; -fx-background-color: #000; -fx-text-fill: #0f0;");

        Button startTrainingBtn = new Button("Start Training");
        startTrainingBtn.setStyle("-fx-background-color: #FF9800; -fx-text-fill: white; -fx-font-size: 14px;");
        startTrainingBtn.setPrefSize(150, 40);

        startTrainingBtn.setOnAction(e -> {
            try {
                // Parse parameters
                String modelName = modelNameField.getText();
                int inputSize = Integer.parseInt(inputSizeField.getText());
                int hiddenLayers = Integer.parseInt(hiddenLayersField.getText());
                String[] hiddenSizesStr = hiddenSizesField.getText().split(",");
                int[] hiddenSizes = new int[hiddenSizesStr.length];
                for (int i = 0; i < hiddenSizesStr.length; i++) {
                    hiddenSizes[i] = Integer.parseInt(hiddenSizesStr[i].trim());
                }
                int outputSize = Integer.parseInt(outputSizeField.getText());
                boolean xavier = xavierCheckBox.isSelected();

                int epochs = Integer.parseInt(epochsField.getText());
                double learningRate = Double.parseDouble(learningRateField.getText());
                int trainSize = Integer.parseInt(trainSizeField.getText());
                int testSize = Integer.parseInt(testSizeField.getText());
                int batchSize = Integer.parseInt(batchSizeField.getText());

                // Disable button during training
                startTrainingBtn.setDisable(true);
                consoleArea.clear();
                consoleArea.appendText("Creating neural network...\n");

                // Create and train model in background thread
                Task<Void> trainingTask = new Task<Void>() {
                    @Override
                    protected Void call() throws Exception {
                        currentModel = new DigitsNN(inputSize, hiddenLayers, hiddenSizes, outputSize, !xavier);

                        Platform.runLater(() -> consoleArea.appendText("Starting training...\n"));

                        // Override the train method to capture output
                        currentModel.trainWithConsole(epochs, learningRate, trainSize, testSize, batchSize,
                                (message) -> Platform.runLater(() -> consoleArea.appendText(message + "\n")));

                        Platform.runLater(() -> {
                            consoleArea.appendText("Training completed!\n");
                            consoleArea.appendText("Saving model...\n");
                        });

                        // Save model
                        ModelSaver.saveModel(currentModel, modelName);

                        Platform.runLater(() -> {
                            consoleArea.appendText("Model saved as: " + modelName + "\n");
                            startTrainingBtn.setDisable(false);
                        });

                        return null;
                    }
                };

                trainingTask.setOnFailed(ex -> {
                    Platform.runLater(() -> {
                        consoleArea.appendText("Error: " + trainingTask.getException().getMessage() + "\n");
                        startTrainingBtn.setDisable(false);
                    });
                });

                new Thread(trainingTask).start();

            } catch (NumberFormatException ex) {
                consoleArea.appendText("Error: Invalid number format\n");
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
                startTrainingBtn
        );

        ScrollPane scrollPane = new ScrollPane(layout);
        scrollPane.setFitToWidth(true);
        Scene scene = new Scene(scrollPane, 500, 700);
        createStage.setScene(scene);
        createStage.show();
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
            alert.setContentText(e.getMessage());
            alert.showAndWait();
        }
    }
}