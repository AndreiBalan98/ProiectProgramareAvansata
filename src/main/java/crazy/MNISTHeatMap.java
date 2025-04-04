package crazy;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Arrays;

public class MNISTHeatMap {

    private final double[][][] heatmaps;
    private final int trainSize;
    private final int testSize;

    public MNISTHeatMap(int trainSize, int testSize) throws IOException {
        this.trainSize = trainSize;
        this.testSize = testSize;
        this.heatmaps = new double[10][28][28];
    }

    public void trainHeatmaps() throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 12345);
        int[] countPerDigit = new int[10];
        Arrays.fill(countPerDigit, 0);

        int trainCount = 0;
        while (trainCount < trainSize) {
            DataSet batch = mnistTrain.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();

            int label = labels.argMax(1).getInt(0);
            double[][] image = features.reshape(28, 28).toDoubleMatrix();

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    heatmaps[label][i][j] += image[i][j];
                }
            }

            countPerDigit[label]++;
            trainCount++;
        }

        for (int digit = 0; digit < 10; digit++) {
            if (countPerDigit[digit] > 0) {
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        heatmaps[digit][i][j] /= countPerDigit[digit];
                    }
                }
            }
        }
    }

    public double computeAccuracy() throws IOException {
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 12345);
        int correctPredictions = 0;
        int totalCount = 0;

        while (totalCount < testSize) {
            DataSet batch = mnistTest.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();

            int actualLabel = labels.argMax(1).getInt(0);
            double[][] image = features.reshape(28, 28).toDoubleMatrix();
            int predictedLabel = predict(image);

            if (predictedLabel == actualLabel) {
                correctPredictions++;
            }
            totalCount++;
        }

        return (double) correctPredictions / testSize;
    }

    private int predict(double[][] image) {
        double maxSimilarity = Double.NEGATIVE_INFINITY;
        int bestMatch = -1;

        for (int digit = 0; digit < 10; digit++) {
            double similarity = 0;

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    similarity += image[i][j] * heatmaps[digit][i][j];
                }
            }

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestMatch = digit;
            }
        }

        return bestMatch;
    }
}