package utils;

import neuralnetwork.DigitsNN;

import java.io.*;
import java.util.function.Function;

public class ModelSaver {

    private static final String MODELS_DIR = "saved_models";

    static {
        // Create models directory if it doesn't exist
        File dir = new File(MODELS_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    public static void saveModel(DigitsNN model, String modelName) throws IOException {
        String fileName = MODELS_DIR + File.separator + modelName + ".model";

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName))) {
            // Save model parameters
            oos.writeInt(model.getInputSize());
            oos.writeInt(model.getNumberOfHiddenLayers());
            oos.writeObject(model.getHiddenLayersSize());
            oos.writeInt(model.getOutputSize());

            // Save weights
            double[][][] weights = model.getWeights();
            oos.writeObject(weights);

            // Save biases
            double[][] biases = model.getBiases();
            oos.writeObject(biases);

            System.out.println("Model saved successfully to: " + fileName);
        }
    }

    public static DigitsNN loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            // Load model parameters
            int inputSize = ois.readInt();
            int numberOfHiddenLayers = ois.readInt();
            int[] hiddenLayersSize = (int[]) ois.readObject();
            int outputSize = ois.readInt();

            // Load weights and biases
            double[][][] weights = (double[][][]) ois.readObject();
            double[][] biases = (double[][]) ois.readObject();

            // Create model with loaded parameters
            DigitsNN model = new DigitsNN(inputSize, numberOfHiddenLayers, hiddenLayersSize, outputSize, true);

            // Set the loaded weights and biases
            model.setWeights(weights);
            model.setBiases(biases);

            System.out.println("Model loaded successfully from: " + filePath);
            return model;
        }
    }

    public static String[] getAvailableModels() {
        File dir = new File(MODELS_DIR);
        String[] files = dir.list((dir1, name) -> name.endsWith(".model"));

        if (files == null) return new String[0];

        // Remove .model extension from names
        for (int i = 0; i < files.length; i++) {
            files[i] = files[i].substring(0, files[i].length() - 6);
        }

        return files;
    }
}