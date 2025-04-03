package utils;

import java.util.Random;

public class Matrix {
    private final double[][] data;

    public Matrix(int rows, int columns) {
        data = new double[rows][columns];
        initializeRandom(0.0, 1.0);
    }

    public Matrix(double[][] data) {
        this.data = data;
    }

    public void initializeRandom(double min, double max) {
        Random rand = new Random();

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = rand.nextDouble() * (max - min) + min;
            }
        }
    }

    public static double[] multiply(double[] vector, double[][] matrix) {
        double[] result = new double[matrix[0].length];

        for (int j = 0; j < matrix[0].length; j++) {
            for (int i = 0; i < vector.length; i++) {
                result[j] += vector[i] * matrix[i][j];
            }
        }

        return result;
    }
}
