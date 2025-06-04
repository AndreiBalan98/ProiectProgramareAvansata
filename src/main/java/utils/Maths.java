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

    public static Double relu(double x) {
        return Math.max(0, x);
    }

    public static Double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    public static Double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static Double sigmoidDerivative(double x) {
        double sigmoidValue = sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
}