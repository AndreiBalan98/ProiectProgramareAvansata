package neuralnetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class ForexNN extends ForwardNeuralNetwork {

    public ForexNN(int inputSize, int numberOfHiddenLayers, int[] hiddenLayersSize, int outputSize, Boolean initializeWith0) {
        super(inputSize, numberOfHiddenLayers, hiddenLayersSize, outputSize, initializeWith0);
    }

    @Override
    public void train(int epochs, double learningRate, int trainSize, int testSize, int batchSize) throws IOException {
        List<double[]> inputList = new ArrayList<>();
        List<double[]> outputList = new ArrayList<>();

        InputStream is = getClass().getClassLoader().getResourceAsStream("forexData.txt");
        if (is == null) {
            throw new IOException("Fisierul forexData.txt nu a fost gasit in folderul resources.");
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        String line;
        int lineNumber = 0;
        int invalidLines = 0;
        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] tokens = line.trim().split("\\s+");

            // Verificăm dacă linia are exact numărul de tokeni necesari
            if (tokens.length != getInputSize() + 2) {
                System.err.println("Linia " + lineNumber + " are un numar incorect de tokeni: " + tokens.length);
                invalidLines++;
                continue;
            }

            double[] inputs = new double[getInputSize()];
            double[] outputs = new double[2];
            boolean validLine = true;

            // Verificăm integritatea datelor pentru intrări
            for (int i = 0; i < getInputSize(); i++) {
                try {
                    inputs[i] = Double.parseDouble(tokens[i]);
                } catch (NumberFormatException e) {
                    System.err.println("Linia " + lineNumber + " are o valoare de input invalida: " + tokens[i]);
                    validLine = false;
                    break;
                }
            }
            if (!validLine) {
                invalidLines++;
                continue;
            }

            // Verificăm integritatea datelor pentru ieșiri
            try {
                outputs[0] = Double.parseDouble(tokens[getInputSize()]);
                outputs[1] = Double.parseDouble(tokens[getInputSize() + 1]);
            } catch (NumberFormatException e) {
                System.err.println("Linia " + lineNumber + " are o valoare de output invalida.");
                invalidLines++;
                continue;
            }

            inputList.add(inputs);
            outputList.add(outputs);
        }
        reader.close();

        if (invalidLines > 0) {
            System.err.println("Au fost gasite " + invalidLines + " linii invalide in fisierul forexData.txt.");
        }

        int totalSamples = inputList.size();
        if (totalSamples < (trainSize + testSize)) {
            throw new IllegalArgumentException("Numarul total de esantioane valide (" + totalSamples
                    + ") este mai mic decat trainSize + testSize (" + (trainSize + testSize) + ").");
        }

        // Convertim listele în matrici
        double[][] datasetInputs = inputList.toArray(new double[0][]);
        double[][] datasetOutputs = outputList.toArray(new double[0][]);

        // Impărțim dataset-ul în seturi de antrenare și testare
        double[][] trainInputs = new double[trainSize][getInputSize()];
        double[][] trainOutputs = new double[trainSize][2];
        double[][] testInputs = new double[testSize][getInputSize()];
        double[][] testOutputs = new double[testSize][2];

        for (int i = 0; i < trainSize; i++) {
            trainInputs[i] = datasetInputs[i];
            trainOutputs[i] = datasetOutputs[i];
        }
        for (int i = 0; i < testSize; i++) {
            testInputs[i] = datasetInputs[trainSize + i];
            testOutputs[i] = datasetOutputs[trainSize + i];
        }

        // Procesul de antrenare
        for (int epoch = 0; epoch < epochs; epoch++) {
            int trainCount = 0;
            for (int i = 0; i < trainSize; i++) {
                backpropagation(trainInputs[i], trainOutputs[i], learningRate);
                trainCount++;

                if (trainCount % batchSize == 0) {
                    double trainAccuracy = computeAccuracy(trainInputs, trainOutputs, trainSize);
                    System.out.println("Epoch " + (epoch + 1) + " - Batch " + trainCount
                            + " - Train Accuracy: " + trainAccuracy);
                }
            }
        }

        double testAccuracy = computeAccuracy(testInputs, testOutputs, testSize);
        System.out.println("Test Accuracy: " + testAccuracy);
    }

    // Metoda de calcul a acurateții folosind softmax pe output
    private double computeAccuracy(double[][] datasetInputs, double[][] datasetOutputs, int dataSize) {
        int correctPredictions = 0;

        for (int i = 0; i < dataSize; i++) {
            double[] rawOutput = feedForward(datasetInputs[i]);

            int predicted0 = (rawOutput[0] > 0.5) ? 1 : 0;
            int predicted1 = (rawOutput[1] > 0.5) ? 1 : 0;

            int actual0 = (datasetOutputs[i][0] > 0.5) ? 1 : 0;
            int actual1 = (datasetOutputs[i][1] > 0.5) ? 1 : 0;

            // Comparam predicția cu valoarea reală
            if (predicted0 == actual0 && predicted1 == actual1) {
                correctPredictions++;
            }
        }

        return (double) correctPredictions / dataSize;
    }
}