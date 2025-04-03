package geneticalgorith;

import neuralnetwork.ForwardNeuralNetwork;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Random;
import java.util.stream.IntStream;

public class Instruments {
    public static void trainNetwork(ForwardNeuralNetwork network, int epochs, double learningRate, int trainSize, int testSize) throws IOException {
        // Încarcă setul de date MNIST
        DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 12345);  // Set de date cu batch size de 1
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 12345);   // Test set cu batch size de 1

        // Împărțirea setului de date pentru antrenament
        int trainCount = 0;
        int testCount = 0;

        // Antrenarea rețelei pe setul de antrenament
        for (int epoch = 0; epoch < epochs; epoch++) {
            int batchCount = 0;
            while (mnistTrain.hasNext() && trainCount < trainSize) {
                DataSet batch = mnistTrain.next();
                INDArray features = batch.getFeatures();
                INDArray labels = batch.getLabels();

                // Reshape și convertește imaginea în vector
                double[] image = features.reshape(28 * 28).toDoubleVector();
                double[] expectedOutput = new double[10];
                expectedOutput[labels.argMax(1).getInt(0)] = 1.0;

                // Aplică backpropagation pentru a ajusta greutățile
                network.backpropagation(image, expectedOutput, learningRate);

                // La fiecare 1000 de batch-uri, afișează scorul curent pe setul de antrenament
                if (batchCount % 1000 == 0) {
                    double trainAccuracy = computeScore(network, mnistTrain, trainSize); // Acuratețea pe setul de antrenament
                    System.out.println("Epoch " + (epoch + 1) + " - Batch " + batchCount + " - Train Accuracy: " + trainAccuracy);
                }

                batchCount++;
                trainCount++;
            }

            // Resetăm iteratorul pentru setul de antrenament
            mnistTrain.reset();
        }

        // Testarea rețelei pe setul de test
        double testAccuracy = computeScore(network, mnistTest, testSize); // Acuratețea pe setul de test
        System.out.println("Test Accuracy: " + testAccuracy);
    }

    public static double computeScore(ForwardNeuralNetwork network, DataSetIterator dataSetIterator, int dataSize) throws IOException {
        int correctPredictions = 0;
        int totalCount = 0;

        while (dataSetIterator.hasNext() && totalCount < dataSize) {
            DataSet batch = dataSetIterator.next();

            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();

            double[] image = features.reshape(28 * 28).toDoubleVector();
            int label = labels.argMax(1).getInt(0);

            double[] output = network.feedForward(image);
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

    public static void transformScore(double[] score) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (double value : score) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }

        if (min == max) {
            max++;
        }

        for (int i = 0; i < score.length; i++) {
            score[i] = (score[i] - min) / (max - min);
        }
    }

    public static double[][][] combineMatrix(double[][][] matrix1, double[][][] matrix2) {
        Random random = new Random();

        double[][][] result = new double[matrix1.length][matrix1[0].length][matrix1[0][0].length];

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[i].length; j++) {
                for (int k = 0; k < matrix1[i][j].length; k++) {
                    if (random.nextDouble() < 0.5) {
                        result[i][j][k] = matrix1[i][j][k];
                    } else {
                        result[i][j][k] = matrix2[i][j][k];
                    }
                }
            }
        }

        return result;
    }


    public static void mutateMatrix(double[][][] matrix, double chanceOfMutation, double maxSizeOfMutation) {
        Random random = new Random();

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                for (int k = 0; k < matrix[i][j].length; k++) {
                    if (random.nextDouble() < chanceOfMutation) {
                        matrix[i][j][k] *= 1 +
                                (random.nextDouble() < 0.5 ?
                                        -random.nextDouble() * maxSizeOfMutation :
                                        random.nextDouble() * maxSizeOfMutation);
                    }
                }
            }
        }
    }
}
