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
    public static double computeScore(ForwardNeuralNetwork network) throws IOException {
        int k = 0, acc = 0;
        DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 12345);

        while (mnistTrain.hasNext() && k < 100) {
            DataSet batch = mnistTrain.next();

            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();

            double[] image = features.reshape(28 * 28).toDoubleVector();
            int label = labels.argMax(1).getInt(0);

            double[] output = network.feedForward(image);
            int outputLabel = IntStream.range(0, output.length)
                    .reduce((i, j) -> output[i] > output[j] ? i : j)
                    .orElse(label);

            if (outputLabel == label) {
                acc++;
            }

            k++;
        }

        return (double)acc / 100;
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
