import java.io.IOException;

import neuralnetwork.ForexNN;
import neuralnetwork.ForwardNeuralNetwork;

public class Main {

    public static void main(String[] args) throws IOException {
        ForwardNeuralNetwork model = new ForexNN(200, 2, new int[]{32, 8}, 2, false);
        model.train(10_000, 0.0001, 800, 200, 100);
    }
}
