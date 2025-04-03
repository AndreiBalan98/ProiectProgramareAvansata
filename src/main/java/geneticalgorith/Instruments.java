package geneticalgorith;

import java.util.Random;

public class Instruments {
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
