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
        DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 12345);

        int trainCount;

        for (int epoch = 0; epoch < epochs; epoch++) {
            trainCount = 0;
            consoleOutput.accept("Starting Epoch " + (epoch + 1) + "/" + epochs);

            while (trainCount < trainSize) {
                DataSet batch = mnistTrain.next();
                INDArray features = batch.getFeatures();
                INDArray labels = batch.getLabels();

                double[] input = features.reshape(28 * 28).toDoubleVector();
                double[] expectedOutput = new double[getOutputSize()];
                expectedOutput[labels.argMax(1).getInt(0)] = 1.0;

                backpropagation(input, expectedOutput, learningRate);

                if ((trainCount + 1) % batchSize == 0) {
                    double trainAccuracy = computeAccuracy(new MnistDataSetIterator(1, true, 12345), Math.min(batchSize * 2, trainSize));
                    String message = String.format("Epoch %d - Batch %d - Train Accuracy: %.4f",
                            (epoch + 1), (trainCount + 1), trainAccuracy);
                    consoleOutput.accept(message);
                }

                trainCount++;
            }

            mnistTrain.reset();
        }

        double testAccuracy = computeAccuracy(mnistTest, testSize);
        consoleOutput.accept(String.format("Training Complete! Test Accuracy: %.4f", testAccuracy));
    }

    private double computeAccuracy(DataSetIterator dataSetIterator, int dataSize) throws IOException {
        int correctPredictions = 0;
        int totalCount = 0;

        while (totalCount < dataSize) {
            DataSet batch = dataSetIterator.next();

            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();

            double[] input = features.reshape(28 * 28).toDoubleVector();
            int label = labels.argMax(1).getInt(0);

            double[] output = feedForward(input);
            int predictedLabel = IntStream.range(0, output.length)
                    .reduce((i, j) -> output[i] > output[j] ? i : j)
                    .orElse(label);

            if (predictedLabel == label) {
                correctPredictions++;
            }

            totalCount++;
        }

        return (double) correctPredictions / dataSize;
    }
}
