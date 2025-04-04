package neuralnetwork;

import java.util.Random;

public class RandomNN extends ForwardNeuralNetwork {

    public RandomNN(int inputSize, int numberOfHiddenLayers, int[] hiddenLayersSize, int outputSize) {
        super(inputSize, numberOfHiddenLayers, hiddenLayersSize, outputSize);
    }

    @Override
    public void train(int epochs, double learningRate, int trainSize, int testSize, int batchSize) {
        int totalSamples = trainSize + testSize;
        double[][] inputs = new double[totalSamples][getInputSize()];
        double[][] outputs = new double[totalSamples][getOutputSize()];
        Random rand = new Random();

        for (int i = 0; i < totalSamples; i++) {
            for (int j = 0; j < getInputSize(); j++) {
                inputs[i][j] = rand.nextDouble() * 10;
            }

            int label = rand.nextInt(getOutputSize());
            for (int k = 0; k < getOutputSize(); k++) {
                outputs[i][k] = (k == label) ? 1.0 : 0.0;
            }
        }

        double[][] trainInputs = new double[trainSize][getInputSize()];
        double[][] trainOutputs = new double[trainSize][getOutputSize()];
        double[][] testInputs = new double[testSize][getInputSize()];
        double[][] testOutputs = new double[testSize][getOutputSize()];
        for (int i = 0; i < trainSize; i++) {
            trainInputs[i] = inputs[i];
            trainOutputs[i] = outputs[i];
        }
        for (int i = 0; i < testSize; i++) {
            testInputs[i] = inputs[trainSize + i];
            testOutputs[i] = outputs[trainSize + i];
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            int trainCount = 0;

            for (int i = 0; i < trainSize; i++) {
                backpropagation(trainInputs[i], trainOutputs[i], learningRate);
                trainCount++;

                if (trainCount % batchSize == 0) {
                    double trainAccuracy = computeAccuracy(trainInputs, trainOutputs, trainSize);
                    System.out.println("Epoch " + (epoch + 1) + " - Batch " + trainCount + " - Train Accuracy: " + trainAccuracy);
                }
            }
        }

        double testAccuracy = computeAccuracy(testInputs, testOutputs, testSize);
        System.out.println("Test Accuracy: " + testAccuracy);
    }

    private double computeAccuracy(double[][] datasetInputs, double[][] datasetOutputs, int dataSize) {
        int correctPredictions = 0;

        for (int i = 0; i < dataSize; i++) {
            double[] output = feedForward(datasetInputs[i]);
            int predictedLabel = argMax(output);
            int actualLabel = argMax(datasetOutputs[i]);
            if (predictedLabel == actualLabel) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / dataSize;
    }

    private int argMax(double[] vector) {
        int index = 0;
        double max = vector[0];

        for (int i = 1; i < vector.length; i++) {
            if (vector[i] > max) {
                max = vector[i];
                index = i;
            }
        }

        return index;
    }
}