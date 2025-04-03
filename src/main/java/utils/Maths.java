package utils;

public class Maths {
    public static double[] matrixMultiplication(double[] vector, double[][] matrix) {
        double[] result = new double[matrix[0].length];

        for (int j = 0; j < matrix[0].length; j++) {
            for (int i = 0; i < vector.length; i++) {
                result[j] += vector[i] * matrix[i][j];
            }
        }

        return result;
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoidValue = sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
}