package neuralnetwork;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import utils.Maths;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

public class DigitsNN extends ForwardNeuralNetwork {

    public DigitsNN(int inputSize, int numberOfHiddenLayers, int[] hiddenLayersSize, int outputSize, Boolean initializeWith0) {
        super(inputSize, numberOfHiddenLayers, hiddenLayersSize, outputSize, initializeWith0);
    }

    public void train(int epochs, double learningRate, int trainSize, int testSize, int batchSize) throws IOException {

        DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 12345);

        int trainCount;

        for (int epoch = 0; epoch < epochs; epoch++) {
            trainCount = 0;

            while (trainCount < trainSize) {
                DataSet batch = mnistTrain.next();
                INDArray features = batch.getFeatures();
                INDArray labels = batch.getLabels();

                double[] input = features.reshape(28 * 28).toDoubleVector();
                double[] expectedOutput = new double[getOutputSize()];
                expectedOutput[labels.argMax(1).getInt(0)] = 1.0;

                backpropagation(input, expectedOutput, learningRate);

                if ((trainCount + 1) % batchSize == 0) {
                    double trainAccuracy = computeAccuracy(new MnistDataSetIterator(1, true, 12345), trainSize);
                    System.out.println("Epoch " + (epoch + 1) + " - Batch " + (trainCount + 1) + " - Train Accuracy: " + trainAccuracy);
                }

                trainCount++;
            }

            mnistTrain.reset();
        }

        double testAccuracy = computeAccuracy(mnistTest, testSize);
        System.out.println("Test Accuracy: " + testAccuracy);
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

    public double[][] getNeuronImage(int layer, int neuron, int label) throws IOException {
        DataSetIterator mnistIter = new MnistDataSetIterator(1, true, 12345);
        DataSet dataSet = null;

        // Caută o imagine cu labelul cerut
        while (mnistIter.hasNext()) {
            DataSet candidate = mnistIter.next();
            int candidateLabel = candidate.getLabels().argMax(1).getInt(0);
            if (candidateLabel == label) {
                dataSet = candidate;
                break;
            }
        }

        if (dataSet == null) {
            throw new IllegalArgumentException("Label not found in MNIST dataset.");
        }

        INDArray features = dataSet.getFeatures();
        double[] input = features.reshape(28 * 28).toDoubleVector();

        // Feedforward + salvare activări
        double[][] activations = new double[getNumberOfHiddenLayers() + 2][];
        activations[0] = input.clone();

        for (int i = 0; i <= getNumberOfHiddenLayers(); i++) {
            activations[i + 1] = Maths.matrixMultiplication(activations[i], getWeights()[i]);
            for (int j = 0; j < activations[i + 1].length; j++) {
                activations[i + 1][j] += getBiases()[i][j];
                activations[i + 1][j] = getActivationFunctions()[i].apply(activations[i + 1][j]);
            }
        }

        // Calcul imagine neuron: weight * activation
        double[] weightsToNeuron = new double[getWeights()[layer].length];
        for (int i = 0; i < weightsToNeuron.length; i++) {
            weightsToNeuron[i] = getWeights()[layer][i][neuron] * activations[layer][i];
        }

        // Normalizare
        double min = Arrays.stream(weightsToNeuron).min().orElse(0);
        double max = Arrays.stream(weightsToNeuron).max().orElse(1);
        double[] normalized = new double[weightsToNeuron.length];
        for (int i = 0; i < weightsToNeuron.length; i++) {
            normalized[i] = (weightsToNeuron[i] - min) / (max - min + 1e-8);
        }

        // Calcul dimensiune pătrat
        int length = normalized.length;
        int size = (int) Math.ceil(Math.sqrt(length)); // dimensiune pătrat

        double[][] image = new double[size][size];
        for (int i = 0; i < length; i++) {
            image[i / size][i % size] = normalized[i];
        }

        return image;
    }

    public double[][][] getNeuronsImages(int layer, int label) throws IOException {
        DataSetIterator mnistIter = new MnistDataSetIterator(1, true, 12345);
        DataSet dataSet = null;

        // Caută o imagine cu labelul cerut
        while (mnistIter.hasNext()) {
            DataSet candidate = mnistIter.next();
            int candidateLabel = candidate.getLabels().argMax(1).getInt(0);
            if (candidateLabel == label) {
                dataSet = candidate;
                break;
            }
        }

        if (dataSet == null) {
            throw new IllegalArgumentException("Label not found in MNIST dataset.");
        }

        INDArray features = dataSet.getFeatures();
        double[] input = features.reshape(28 * 28).toDoubleVector();

        // Feedforward + salvare activări
        double[][] activations = new double[getNumberOfHiddenLayers() + 2][];
        activations[0] = input.clone();

        for (int i = 0; i <= getNumberOfHiddenLayers(); i++) {
            activations[i + 1] = Maths.matrixMultiplication(activations[i], getWeights()[i]);
            for (int j = 0; j < activations[i + 1].length; j++) {
                activations[i + 1][j] += getBiases()[i][j];
                activations[i + 1][j] = getActivationFunctions()[i].apply(activations[i + 1][j]);
            }
        }

        // Obține numărul de neuroni pe stratul specificat
        int neuronCount = getWeights()[layer][0].length;
        double[][][] images = new double[neuronCount][][];

        for (int neuron = 0; neuron < neuronCount; neuron++) {
            double[] weightsToNeuron = new double[getWeights()[layer].length];
            for (int i = 0; i < weightsToNeuron.length; i++) {
                weightsToNeuron[i] = getWeights()[layer][i][neuron] * activations[layer][i];
            }

            // Normalizare
            double min = Arrays.stream(weightsToNeuron).min().orElse(0);
            double max = Arrays.stream(weightsToNeuron).max().orElse(1);
            double[] normalized = new double[weightsToNeuron.length];
            for (int i = 0; i < weightsToNeuron.length; i++) {
                normalized[i] = (weightsToNeuron[i] - min) / (max - min + 1e-8);
            }

            // Construire imagine pătrată
            int length = normalized.length;
            int size = (int) Math.ceil(Math.sqrt(length));

            double[][] image = new double[size][size];
            for (int i = 0; i < length; i++) {
                image[i / size][i % size] = normalized[i];
            }

            images[neuron] = image;
        }

        return images;
    }
}
