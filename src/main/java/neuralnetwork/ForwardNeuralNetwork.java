package neuralnetwork;

import utils.Maths;
import utils.Matrix;

import java.util.Arrays;
import java.util.Random;

//TODO: IMPLEMENT BIAS

public class ForwardNeuralNetwork {
    private final int inputSize;
    private final int numberOfHiddenLayers;
    private final int[] hiddenLayersSize;
    private final int outputSize;

    private double[][][] neuralNetwork;

    public ForwardNeuralNetwork(double[][][] neuralNetwork) {
        this.inputSize = neuralNetwork[0].length;
        this.numberOfHiddenLayers = neuralNetwork.length - 1;
        this.hiddenLayersSize = new int[numberOfHiddenLayers];
        this.outputSize = neuralNetwork[neuralNetwork.length -1][0].length;
        this.neuralNetwork = neuralNetwork;

        for (int i = 0; i < numberOfHiddenLayers; i++) {
            hiddenLayersSize[i] = neuralNetwork[i][0].length;
        }
    }

    public ForwardNeuralNetwork(int inputSize, int numberOfHiddenLayers, int[]hiddenLayersSize, int outputSize) {
        this.inputSize = inputSize;
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.hiddenLayersSize = hiddenLayersSize;
        this.outputSize = outputSize;

        generateNeuralNetwork();
        initializeRandom(0.0, 1.0);
    }

    private void generateNeuralNetwork() {
        neuralNetwork = new double[numberOfHiddenLayers + 1][][];
        for (int i = 0; i < numberOfHiddenLayers + 1; i++) {
            if (i == 0) {
                neuralNetwork[i] = new double[inputSize][hiddenLayersSize[0]];
            }
            else if (i == numberOfHiddenLayers) {
                neuralNetwork[i] = new double[hiddenLayersSize[numberOfHiddenLayers - 1]][outputSize];
            }
            else {
                neuralNetwork[i] = new double[hiddenLayersSize[i - 1]][hiddenLayersSize[i]];
            }
        }
    }

    public void initializeRandom(double min, double max) {
        Random rand = new Random();

        for (int i = 0; i < numberOfHiddenLayers + 1; i++) {
            for (int j = 0; j < neuralNetwork[i].length; j++) {
                for (int k = 0; k < neuralNetwork[i][j].length; k++) {
                    neuralNetwork[i][j][k] = min + rand.nextDouble() * (max - min);
                }
            }
        }
    }

    public double[] feedForward(double[] input) {
        double[] train = input.clone();

        for (int i = 0; i < numberOfHiddenLayers + 1; i++) {
            train = Matrix.multiply(train, neuralNetwork[i]);
            train = Arrays.stream(train).map(Maths::sigmoid).toArray();
        }

        return train;
    }

    public void backpropagation(double[] input, double[] expectedOutput, double learningRate) {
        // 1. Forward pass - salvăm activările
        double[][] activations = new double[numberOfHiddenLayers + 2][];
        activations[0] = input.clone();

        for (int i = 0; i < numberOfHiddenLayers + 1; i++) {
            activations[i + 1] = Matrix.multiply(activations[i], neuralNetwork[i]);
            activations[i + 1] = Arrays.stream(activations[i + 1]).map(Maths::sigmoid).toArray();
        }

        // 2. Calculăm eroarea pentru stratul de ieșire
        double[][] deltas = new double[numberOfHiddenLayers + 1][];
        deltas[numberOfHiddenLayers] = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double output = activations[numberOfHiddenLayers + 1][i];
            deltas[numberOfHiddenLayers][i] = (output - expectedOutput[i]) * Maths.sigmoidDerivative(output);
        }

        // 3. Backward pass - calculăm deltele pentru straturile ascunse
        for (int l = numberOfHiddenLayers - 1; l >= 0; l--) {
            deltas[l] = new double[hiddenLayersSize[l]];

            for (int j = 0; j < hiddenLayersSize[l]; j++) {
                double sum = 0;
                for (int k = 0; k < deltas[l + 1].length; k++) {
                    sum += neuralNetwork[l + 1][j][k] * deltas[l + 1][k];
                }
                deltas[l][j] = sum * Maths.sigmoidDerivative(activations[l + 1][j]);
            }
        }

        // 4. Actualizăm ponderile
        for (int l = 0; l < numberOfHiddenLayers + 1; l++) {
            for (int j = 0; j < neuralNetwork[l].length; j++) {
                for (int k = 0; k < neuralNetwork[l][j].length; k++) {
                    neuralNetwork[l][j][k] -= learningRate * deltas[l][k] * activations[l][j];
                }
            }
        }
    }

    public double[][][] getNeuralNetwork() {
        return neuralNetwork;
    }
}
