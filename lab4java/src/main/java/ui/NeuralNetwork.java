package ui;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    private final List<Layer> layers;
    private final int inputDim;
    private final String architecture;

    public NeuralNetwork(int inputDim, String architecture) {
        this.inputDim = inputDim;
        this.architecture = architecture;
        this.layers = new ArrayList<>();
        initializeNetwork();
    }

    private void initializeNetwork() {
        String[] parts = architecture.split("s");
        int[] hiddenDims = new int[parts.length];

        for (int i = 0; i < parts.length; i++) {
            hiddenDims[i] = Integer.parseInt(parts[i]);
        }

        layers.add(new Layer(inputDim, hiddenDims[0], true));

        for (int i = 1; i < hiddenDims.length; i++) {
            layers.add(new Layer(hiddenDims[i-1], hiddenDims[i], true));
        }

        layers.add(new Layer(hiddenDims[hiddenDims.length-1], 1, false));
    }

    public double predict(double[] input) {
        double[] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output[0];
    }

    public double calculateMSE(List<Double[]> dataset) {
        double sumSquaredErrors = 0.0;

        for (Double[] instance : dataset) {
            double[] input = new double[inputDim];
            double target = instance[inputDim];

            for (int i = 0; i < inputDim; i++) {
                input[i] = instance[i];
            }

            double prediction = predict(input);
            double error = target - prediction;
            sumSquaredErrors += error * error;
        }

        return sumSquaredErrors / dataset.size();
    }

    public List<Double> getFlattenedWeights() {
        List<Double> allWeights = new ArrayList<>();

        for (Layer layer : layers) {
            for (double[] neuronWeights : layer.weights) {
                for (double weight : neuronWeights) {
                    allWeights.add(weight);
                }
            }
            for (double bias : layer.biases) {
                allWeights.add(bias);
            }
        }

        return allWeights;
    }

    public void setWeightsFromFlattened(List<Double> weights) {
        int index = 0;

        for (Layer layer : layers) {
            for (int i = 0; i < layer.weights.length; i++) {
                for (int j = 0; j < layer.weights[i].length; j++) {
                    layer.weights[i][j] = weights.get(index++);
                }
            }

            for (int i = 0; i < layer.biases.length; i++) {
                layer.biases[i] = weights.get(index++);
            }
        }
    }

    public NeuralNetwork copy() {
        NeuralNetwork copy = new NeuralNetwork(inputDim, architecture);
        copy.setWeightsFromFlattened(this.getFlattenedWeights());
        return copy;
    }

    private static class Layer {
        final double[][] weights;
        final double[] biases;
        final boolean hasActivation;

        Layer(int inputSize, int outputSize, boolean hasActivation) {
            this.weights = new double[outputSize][inputSize];
            this.biases = new double[outputSize];
            this.hasActivation = hasActivation;
            initializeParameters();
        }

        private void initializeParameters() {
            Random rand = new Random();
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = rand.nextGaussian() * 0.01;
                }
                biases[i] = rand.nextGaussian() * 0.01;
            }
        }

        double[] forward(double[] input) {
            double[] output = new double[weights.length];

            for (int i = 0; i < weights.length; i++) {
                output[i] = biases[i];
                for (int j = 0; j < weights[i].length; j++) {
                    output[i] += weights[i][j] * input[j];
                }

                if (hasActivation) {
                    output[i] = 1.0 / (1.0 + Math.exp(-output[i]));
                }
            }

            return output;
        }
    }
}