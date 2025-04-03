import geneticalgorith.Instruments;
import neuralnetwork.ForwardNeuralNetwork;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import javax.sound.midi.Instrument;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) throws IOException {
        int populationSize = 10;
        ArrayList<ForwardNeuralNetwork> population = new ArrayList<>();
        double[] score = new double[populationSize];

        for (int i = 0; i < populationSize; i++) {
            population.add(new ForwardNeuralNetwork(784, 3, new int[]{500, 500, 500}, 10));
        }

        for (int k = 0; k < 1_000; k++) {
            for (int i = 0; i < populationSize; i++) {
                score[i] = computeScore(population.get(i));
            }

            System.out.println("Generation " + k + ": " + Arrays.stream(score).max().getAsDouble());

            ArrayList<ForwardNeuralNetwork> nextPopulation = new ArrayList<>();
            transformScore(score);
            Random rand = new Random();

            for (int i = 0; i < populationSize; i++) {
                int a, b, k2 = 0;

                do {
                    a = rand.nextInt(populationSize);
                    k2++;
                } while (!(rand.nextDouble() < score[a]) && k2 < 100);

                k2 = 0;

                do {
                    b = rand.nextInt(populationSize);
                    k2++;
                } while (!(rand.nextDouble() < score[b]) && k2 < 100);

                double[][][] temp = Instruments.combineMatrix(population.get(a).getNeuralNetwork(), population.get(b).getNeuralNetwork());
                Instruments.mutateMatrix(temp, 0.1, 10.0);

                nextPopulation.add(new ForwardNeuralNetwork(temp));
            }

            population = nextPopulation;
        }
    }

    private static double computeScore(ForwardNeuralNetwork network) throws IOException {
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

    private static void transformScore(double[] score) {
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
}

/*DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 12345); // Batch size 1 pentru imagini individuale

        while (mnistTrain.hasNext()) {
DataSet batch = mnistTrain.next();

INDArray features = batch.getFeatures(); // Imagini 28x28
INDArray labels = batch.getLabels(); // Etichete

// Convertim imaginea 28x28 din vector unidimensional
double[][] image = features.reshape(28, 28).toDoubleMatrix();
int label = labels.argMax(1).getInt(0); // Obținem eticheta

// Afișăm imaginea și eticheta asociată
            System.out.println("Eticheta: " + label);
            for (double[] row : image) {
        for (double pixel : row) {
        System.out.print(pixel > 0.5 ? "#" : "."); // Convertim pixelii pentru afișare
                }
                        System.out.println();
            }
                    System.out.println("----------------------");
        }*/
