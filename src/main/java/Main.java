import java.io.IOException;

import neuralnetwork.DigitsNN;
import neuralnetwork.ForwardNeuralNetwork;

public class Main {
    public static void main(String[] args) throws IOException {

        ForwardNeuralNetwork model = new DigitsNN(784, 2, new int[]{32, 16}, 10);
        model.train(5, 0.1, 10_000, 10_000, 1_000);
        //new Application(network);
    }
}
