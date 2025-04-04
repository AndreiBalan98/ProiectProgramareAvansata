package crazy;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public class MNISTHeatMapPro {
    private final double[][][] heatmaps;
    private final int trainSize;
    private final int testSize;

    public MNISTHeatMapPro(int trainSize, int testSize) throws IOException {
        this.trainSize = trainSize;
        this.testSize = testSize;
        this.heatmaps = new double[10][28][28];
    }

    public void trainHeatmaps() throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 12345);

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

            trainCount++;
        }
    }

    public void applyGaussianBlur() {
        double sigma = 1.0;
        double[][] kernel = generateGaussianKernel(3, sigma);

        for (int digit = 0; digit < 10; digit++) {
            heatmaps[digit] = convolve(heatmaps[digit], kernel);
        }
    }

    private double[][] generateGaussianKernel(int size, double sigma) {
        double[][] kernel = new double[size][size];
        double sum = 0.0;
        int halfSize = size / 2;

        for (int i = -halfSize; i <= halfSize; i++) {
            for (int j = -halfSize; j <= halfSize; j++) {
                kernel[i + halfSize][j + halfSize] = Math.exp(-(i * i + j * j) / (2 * sigma * sigma));
                sum += kernel[i + halfSize][j + halfSize];
            }
        }

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                kernel[i][j] /= sum;
            }
        }
        return kernel;
    }

    private double[][] convolve(double[][] image, double[][] kernel) {
        int size = kernel.length;
        int halfSize = size / 2;
        double[][] result = new double[28][28];

        for (int i = halfSize; i < 28 - halfSize; i++) {
            for (int j = halfSize; j < 28 - halfSize; j++) {
                double sum = 0.0;
                for (int ki = -halfSize; ki <= halfSize; ki++) {
                    for (int kj = -halfSize; kj <= halfSize; kj++) {
                        sum += image[i + ki][j + kj] * kernel[ki + halfSize][kj + halfSize];
                    }
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    public void normalizeHeatmaps() {
        for (int digit = 0; digit < 10; digit++) {
            double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    min = Math.min(min, heatmaps[digit][i][j]);
                    max = Math.max(max, heatmaps[digit][i][j]);
                }
            }
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    heatmaps[digit][i][j] = (heatmaps[digit][i][j] - min) / (max - min + 1e-8);
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