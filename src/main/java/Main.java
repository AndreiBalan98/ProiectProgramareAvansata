import geneticalgorith.Instruments;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import java.io.IOException;
import java.util.function.Function;
import neuralnetwork.ForwardNeuralNetwork;
import ui.Application;

public class Main {

    public static void main(String[] args) throws IOException { //Wrote from GitHub
        int inputSize = 28 * 28;
        int[] hiddenLayersSize = {128, 64}; // Două straturi ascunse
        int outputSize = 10;
        int epochs = 5;
        double learningRate = 0.01;

        ForwardNeuralNetwork network = new ForwardNeuralNetwork(inputSize, hiddenLayersSize.length, hiddenLayersSize, outputSize);

        // Apelează funcția care face training-ul
        Instruments.trainNetwork(network, epochs, learningRate, 10_000, 5_000);
        new Application(network);
    }
}
