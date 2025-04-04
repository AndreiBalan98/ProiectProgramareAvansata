package neuralnetwork;

import utils.Maths;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public abstract class ForwardNeuralNetwork {

    private final int inputSize;
    private final int numberOfHiddenLayers;
    private final int[] hiddenLayersSize;
    private final int outputSize;
    private double[][][] weights;
    private double[][] biases;
    private final Function<Double, Double>[] activationFunctions;
    private final Function<Double, Double>[] activationDerivatives;

    public ForwardNeuralNetwork(int inputSize, int numberOfHiddenLayers, int[] hiddenLayersSize, int outputSize, Boolean initializeWith0) {
        this.inputSize = inputSize;
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.hiddenLayersSize = hiddenLayersSize;
        this.outputSize = outputSize;

        this.activationFunctions = new Function[numberOfHiddenLayers + 1];
        this.activationDerivatives = new Function[numberOfHiddenLayers + 1];

        for (int i = 0; i < numberOfHiddenLayers; i++) {
            this.activationFunctions[i] = Maths::relu;
            this.activationDerivatives[i] = Maths::reluDerivative;
        }

        this.activationFunctions[numberOfHiddenLayers] = Maths::sigmoid;
        this.activationDerivatives[numberOfHiddenLayers] = Maths::sigmoidDerivative;

        initializeNetwork();
        if (!initializeWith0) {
            initializeWeightsXavier();
        }
    }

    private void initializeNetwork() {
        weights = new double[numberOfHiddenLayers + 1][][];
        biases = new double[numberOfHiddenLayers + 1][];

        for (int i = 0; i <= numberOfHiddenLayers; i++) {
            int inputLayerSize = (i == 0) ? inputSize : hiddenLayersSize[i - 1];
            int outputLayerSize = (i == numberOfHiddenLayers) ? outputSize : hiddenLayersSize[i];

            weights[i] = new double[inputLayerSize][outputLayerSize];
            biases[i] = new double[outputLayerSize];
        }
    }

    private void initializeWeightsXavier() {
        Random rand = new Random();

        for (int i = 0; i <= numberOfHiddenLayers; i++) {
            int inputLayerSize = weights[i].length;
            int outputLayerSize = weights[i][0].length;
            double limit = Math.sqrt(6.0 / (inputLayerSize + outputLayerSize));

            for (int j = 0; j < inputLayerSize; j++) {
                for (int k = 0; k < outputLayerSize; k++) {
                    weights[i][j][k] = (rand.nextDouble() * 2 - 1) * limit;
                }
            }

            Arrays.fill(biases[i], 0.0);
        }
    }

    public double[] feedForward(double[] input) {
        double[] activations = input.clone();

        for (int i = 0; i <= numberOfHiddenLayers; i++) {
            activations = Maths.matrixMultiplication(activations, weights[i]);

            for (int j = 0; j < activations.length; j++) {
                activations[j] += biases[i][j];
                activations[j] = activationFunctions[i].apply(activations[j]);
            }
        }

        return activations;
    }

    //TODO: I MUST UNDERSTAND THIS METHOD IN DEPTH
    public void backpropagation(double[] input, double[] expectedOutput, double learningRate) {
        double[][] activations = new double[numberOfHiddenLayers + 2][];
        activations[0] = input.clone();

        for (int i = 0; i <= numberOfHiddenLayers; i++) {
            activations[i + 1] = Maths.matrixMultiplication(activations[i], weights[i]);

            for (int j = 0; j < activations[i + 1].length; j++) {
                activations[i + 1][j] += biases[i][j];
                activations[i + 1][j] = activationFunctions[i].apply(activations[i + 1][j]);
            }
        }

        double[][] deltas = new double[numberOfHiddenLayers + 1][];
        deltas[numberOfHiddenLayers] = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double output = activations[numberOfHiddenLayers + 1][i];
            deltas[numberOfHiddenLayers][i] = (output - expectedOutput[i]) * activationDerivatives[numberOfHiddenLayers].apply(output);
        }

        for (int l = numberOfHiddenLayers - 1; l >= 0; l--) {
            deltas[l] = new double[hiddenLayersSize[l]];

            for (int j = 0; j < hiddenLayersSize[l]; j++) {
                double sum = 0;

                for (int k = 0; k < deltas[l + 1].length; k++) {
                    sum += weights[l + 1][j][k] * deltas[l + 1][k];
                }

                deltas[l][j] = sum * activationDerivatives[l].apply(activations[l + 1][j]);
            }
        }

        for (int l = 0; l <= numberOfHiddenLayers; l++) {
            for (int j = 0; j < weights[l].length; j++) {
                for (int k = 0; k < weights[l][j].length; k++) {
                    weights[l][j][k] -= learningRate * deltas[l][k] * activations[l][j];
                }
            }

            for (int j = 0; j < biases[l].length; j++) {
                biases[l][j] -= learningRate * deltas[l][j];
            }
        }
    }

    public abstract void train(int epochs, double learningRate, int trainSize, int testSize, int batchSize) throws IOException;

    public int getInputSize() {
        return inputSize;
    }

    public int getNumberOfHiddenLayers() {
        return numberOfHiddenLayers;
    }

    public int[] getHiddenLayersSize() {
        return hiddenLayersSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public double[][][] getWeights() {
        return weights;
    }

    public double[][] getBiases() {
        return biases;
    }

    public Function<Double, Double>[] getActivationFunctions() {
        return activationFunctions;
    }

    public Function<Double, Double>[] getActivationDerivatives() {
        return activationDerivatives;
    }
}