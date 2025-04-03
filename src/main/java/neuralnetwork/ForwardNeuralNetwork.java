package neuralnetwork;

import utils.Maths;
import utils.Matrix;

import java.util.Arrays;
import java.util.Random;

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
        initializeRandom(0.0, 0.01);
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

    public double[][][] getNeuralNetwork() {
        return neuralNetwork;
    }
}
