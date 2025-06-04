import neuralnetwork.DigitsNN;
import userinterface.DigitsApplication1;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        DigitsNN model = new DigitsNN(784, 3, new int[]{64, 32, 16}, 10, Boolean.FALSE);
        model.train(5, 0.01, 10_000, 10_000, 1_000);
        new DigitsApplication1(model);
    }
}
