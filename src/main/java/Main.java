import java.io.IOException;

import crazy.MNISTHeatMap;
import crazy.MNISTHeatMapPro;
import neuralnetwork.DigitsNN;
import neuralnetwork.ForwardNeuralNetwork;

import neuralnetwork.RandomNN;
import userinterface.Application;

public class Main {

    public static void main(String[] args) throws IOException {

        MNISTHeatMap heatModel = new MNISTHeatMap(60_000, 10_000);
        heatModel.trainHeatmaps();
        System.out.println("Accuracy MNISTHeatMap: " + heatModel.computeAccuracy());

        MNISTHeatMapPro proHeatModel = new MNISTHeatMapPro(60_000, 10_000);
        proHeatModel.trainHeatmaps();
        proHeatModel.applyGaussianBlur();
        proHeatModel.normalizeHeatmaps();
        System.out.println("Accuracy MNISTHeatMapPro: " + proHeatModel.computeAccuracy());
    }
}
