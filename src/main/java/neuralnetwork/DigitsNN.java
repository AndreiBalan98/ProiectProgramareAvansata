package neuralnetwork;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import utils.Maths;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class DigitsNN extends ForwardNeuralNetwork {

    public DigitsNN(int inputSize, int numberOfHiddenLayers, int[] hiddenLayersSize, int outputSize, Boolean initializeWith0) {
        super(inputSize, numberOfHiddenLayers, hiddenLayersSize, outputSize, initializeWith0);
    }

    @Override
    public void train(int epochs, double learningRate, int trainSize, int testSize, int batchSize) throws IOException {
        trainWithConsole(epochs, learningRate, trainSize, testSize, batchSize, System.out::println);
    }

    public void trainWithConsole(int epochs, double learningRate, int trainSize, int testSize, int batchSize, Consumer<String> consoleOutput) throws IOException {
        DataSetIterator mnistTrain = null;
        DataSetIterator mnistTest = null;

        try {
            consoleOutput.accept("Initializing MNIST dataset...");

            // Create MNIST iterators with better error handling
            mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
            mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

            consoleOutput.accept("MNIST dataset loaded successfully!");

        } catch (Exception e) {
            String errorMsg = "Failed to load MNIST dataset: " +
                    (e.getMessage() != null ? e.getMessage() : "Unknown error");
            consoleOutput.accept(errorMsg);
            consoleOutput.accept("This might be due to network issues or missing dependencies.");
            consoleOutput.accept("Stack trace: " + e.getClass().getSimpleName());
            if (e.getCause() != null) {
                consoleOutput.accept("Caused by: " + e.getCause().getMessage());
            }
            throw new IOException(errorMsg, e);
        }

        int trainCount = 0;
        int totalBatches = (trainSize + batchSize - 1) / batchSize; // Ceiling division

        try {
            for (int epoch = 0; epoch < epochs; epoch++) {
                trainCount = 0;
                int currentBatch = 0;
                consoleOutput.accept("Starting Epoch " + (epoch + 1) + "/" + epochs);

                // Reset iterator at the beginning of each epoch
                mnistTrain.reset();

                while (trainCount < trainSize && mnistTrain.hasNext()) {
                    try {
                        DataSet batch = mnistTrain.next();

                        if (batch == null) {
                            consoleOutput.accept("Warning: Received null batch, skipping...");
                            continue;
                        }

                        INDArray features = batch.getFeatures();
                        INDArray labels = batch.getLabels();

                        if (features == null || labels == null) {
                            consoleOutput.accept("Warning: Received batch with null features or labels, skipping...");
                            continue;
                        }

                        // Process each example in the batch
                        int batchActualSize = Math.min(batchSize, (int)features.size(0));

                        for (int i = 0; i < batchActualSize && trainCount < trainSize; i++) {
                            INDArray singleFeature = features.getRow(i);
                            INDArray singleLabel = labels.getRow(i);

                            double[] input = singleFeature.reshape(28 * 28).toDoubleVector();
                            double[] expectedOutput = new double[getOutputSize()];

                            int labelIndex = singleLabel.argMax(1).getInt(0);
                            if (labelIndex >= 0 && labelIndex < getOutputSize()) {
                                expectedOutput[labelIndex] = 1.0;
                            }

                            backpropagation(input, expectedOutput, learningRate);
                            trainCount++;
                        }

                        currentBatch++;

                        // Report progress every few batches
                        if (currentBatch % Math.max(1, totalBatches / 10) == 0) {
                            double progress = (double) trainCount / trainSize * 100;
                            consoleOutput.accept(String.format("Epoch %d - Progress: %.1f%% (%d/%d samples)",
                                    (epoch + 1), progress, trainCount, trainSize));
                        }

                    } catch (Exception e) {
                        String batchError = "Error processing batch " + currentBatch + ": " +
                                (e.getMessage() != null ? e.getMessage() : "Unknown batch error");
                        consoleOutput.accept(batchError);
                        consoleOutput.accept("Continuing with next batch...");
                        // Continue with the next batch instead of failing completely
                    }
                }

                // Compute and report epoch accuracy
                try {
                    mnistTest.reset();
                    double testAccuracy = computeAccuracy(mnistTest, Math.min(testSize, 1000));
                    consoleOutput.accept(String.format("Epoch %d completed - Test Accuracy: %.4f",
                            (epoch + 1), testAccuracy));
                } catch (Exception e) {
                    consoleOutput.accept("Warning: Could not compute test accuracy: " +
                            (e.getMessage() != null ? e.getMessage() : "Unknown error"));
                }
            }

            // Final test accuracy
            try {
                mnistTest.reset();
                double finalTestAccuracy = computeAccuracy(mnistTest, testSize);
                consoleOutput.accept(String.format("Training Complete! Final Test Accuracy: %.4f", finalTestAccuracy));
            } catch (Exception e) {
                consoleOutput.accept("Training completed, but could not compute final test accuracy: " +
                        (e.getMessage() != null ? e.getMessage() : "Unknown error"));
            }

        } catch (Exception e) {
            String trainError = "Training failed: " + (e.getMessage() != null ? e.getMessage() : "Unknown training error");
            consoleOutput.accept(trainError);
            throw new IOException(trainError, e);
        }
    }

    private double computeAccuracy(DataSetIterator dataSetIterator, int dataSize) throws IOException {
        int correctPredictions = 0;
        int totalCount = 0;
        int batchCount = 0;

        try {
            while (totalCount < dataSize && dataSetIterator.hasNext()) {
                DataSet batch = dataSetIterator.next();

                if (batch == null) continue;

                INDArray features = batch.getFeatures();
                INDArray labels = batch.getLabels();

                if (features == null || labels == null) continue;

                int batchSize = Math.min((int)features.size(0), dataSize - totalCount);

                for (int i = 0; i < batchSize; i++) {
                    try {
                        INDArray singleFeature = features.getRow(i);
                        INDArray singleLabel = labels.getRow(i);

                        double[] input = singleFeature.reshape(28 * 28).toDoubleVector();
                        int label = singleLabel.argMax(1).getInt(0);

                        double[] output = feedForward(input);
                        int predictedLabel = IntStream.range(0, output.length)
                                .reduce((j, k) -> output[j] > output[k] ? j : k)
                                .orElse(0);

                        if (predictedLabel == label) {
                            correctPredictions++;
                        }

                        totalCount++;

                    } catch (Exception e) {
                        // Skip this sample and continue
                        totalCount++;
                    }
                }

                batchCount++;
            }
        } catch (Exception e) {
            throw new IOException("Error computing accuracy: " +
                    (e.getMessage() != null ? e.getMessage() : "Unknown accuracy error"), e);
        }

        if (totalCount == 0) {
            throw new IOException("No valid samples found for accuracy computation");
        }

        return (double) correctPredictions / totalCount;
    }
}