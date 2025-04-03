package neuralnetwork;

import utils.Maths;
import utils.Matrix;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class ForwardNeuralNetwork {
    private final int inputSize;
    private final int numberOfHiddenLayers;
    private final int[] hiddenLayersSize;
    private final int outputSize;
    private double[][][] weights;
    private double[][] biases;
    private final Function<Double, Double>[] activationFunctions;
    private final Function<Double, Double>[] activationDerivatives;

    // Constructorul modificat
    public ForwardNeuralNetwork(int inputSize, int numberOfHiddenLayers, int[] hiddenLayersSize, int outputSize) {
        this.inputSize = inputSize;
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.hiddenLayersSize = hiddenLayersSize;
        this.outputSize = outputSize;

        // Folosirea funcțiilor implicite dacă nu sunt furnizate
        this.activationFunctions = new Function[numberOfHiddenLayers + 1];
        this.activationDerivatives = new Function[numberOfHiddenLayers + 1];

        // Funcțiile de activare implicite pentru straturile ascunse și stratul de ieșire
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            this.activationFunctions[i] = this::relu;
            this.activationDerivatives[i] = this::reluDerivative;
        }

        // Funcția de activare pentru stratul de ieșire (pentru exemplu: Sigmoid)
        this.activationFunctions[numberOfHiddenLayers] = this::sigmoid;
        this.activationDerivatives[numberOfHiddenLayers] = this::sigmoidDerivative;

        initializeNetwork();
        initializeWeightsXavier();
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

    // Funcțiile de activare implicite
    private double relu(double x) {
        return Math.max(0, x);  // ReLU
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;  // Derivata ReLU
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));  // Sigmoid
    }

    private double sigmoidDerivative(double x) {
        double sigmoidValue = sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);  // Derivata Sigmoid
    }

    public double[] feedForward(double[] input) {
        double[] activations = input.clone();

        for (int i = 0; i <= numberOfHiddenLayers; i++) {
            activations = Matrix.multiply(activations, weights[i]);
            for (int j = 0; j < activations.length; j++) {
                activations[j] += biases[i][j];
                activations[j] = activationFunctions[i].apply(activations[j]);
            }
        }
        return activations;
    }

    public void backpropagation(double[] input, double[] expectedOutput, double learningRate) {
        double[][] activations = new double[numberOfHiddenLayers + 2][];
        activations[0] = input.clone();

        for (int i = 0; i <= numberOfHiddenLayers; i++) {
            activations[i + 1] = Matrix.multiply(activations[i], weights[i]);
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

    public double[][][] getWeights() {
        return weights;
    }

    public double[][] getBiases() {
        return biases;
    }
}