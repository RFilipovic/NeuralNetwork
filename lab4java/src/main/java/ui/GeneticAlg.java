package ui;

import java.util.*;

public class GeneticAlg {
    private final int inputDim;
    private final String architecture;
    private final int popsize;
    private final int elitism;
    private final double p;
    private final double K;
    private final int iter;
    private final Random random;

    public GeneticAlg(int inputDim, String architecture, int popsize, int elitism, double p, double K, int iter) {
        this.inputDim = inputDim;
        this.architecture = architecture;
        this.popsize = popsize;
        this.elitism = elitism;
        this.p = p;
        this.K = K;
        this.iter = iter;
        this.random = new Random();
    }

    public NeuralNetwork evolve(List<Double[]> trainSet) {
        List<Individual> population = initializePopulation();

        for (int generation = 1; generation <= iter; generation++) {
            evaluatePopulation(population, trainSet);
            population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

            if (generation % 2000 == 0) {
                double trainError = (1.0 / population.get(0).fitness) - 1.0;
                System.out.printf("[Train error @%d]: %.6f\n", generation, trainError);
            }

            if (generation < iter) {
                population = createNextGeneration(population);
            }
        }

        evaluatePopulation(population, trainSet);
        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
        return population.get(0).network;
    }

    private List<Individual> initializePopulation() {
        List<Individual> population = new ArrayList<>();
        for (int i = 0; i < popsize; i++) {
            NeuralNetwork network = new NeuralNetwork(inputDim, architecture);
            population.add(new Individual(network));
        }
        return population;
    }

    private void evaluatePopulation(List<Individual> population, List<Double[]> trainSet) {
        for (Individual individual : population) {
            double mse = individual.network.calculateMSE(trainSet);
            individual.fitness = 1.0 / (1.0 + mse);
        }
    }

    private List<Individual> createNextGeneration(List<Individual> population) {
        List<Individual> nextGeneration = new ArrayList<>();

        for (int i = 0; i < elitism; i++) {
            nextGeneration.add(new Individual(population.get(i).network.copy()));
        }

        while (nextGeneration.size() < popsize) {
            Individual parent1 = selectParent(population);
            Individual parent2 = selectParent(population);
            Individual child = crossover(parent1, parent2);
            mutate(child);
            nextGeneration.add(child);
        }

        return nextGeneration;
    }

    private Individual selectParent(List<Individual> population) {
        double totalFitness = 0;
        for (Individual individual : population) {
            totalFitness += individual.fitness;
        }

        double randomValue = random.nextDouble() * totalFitness;
        double cumulativeFitness = 0;

        for (Individual individual : population) {
            cumulativeFitness += individual.fitness;
            if (cumulativeFitness >= randomValue) {
                return individual;
            }
        }
        return population.get(population.size() - 1);
    }

    private Individual crossover(Individual parent1, Individual parent2) {
        List<Double> weights1 = parent1.network.getFlattenedWeights();
        List<Double> weights2 = parent2.network.getFlattenedWeights();
        List<Double> childWeights = new ArrayList<>();

        for (int i = 0; i < weights1.size(); i++) {
            double averageWeight = (weights1.get(i) + weights2.get(i)) / 2.0;
            childWeights.add(averageWeight);
        }

        NeuralNetwork childNetwork = new NeuralNetwork(inputDim, architecture);
        childNetwork.setWeightsFromFlattened(childWeights);
        return new Individual(childNetwork);
    }

    private void mutate(Individual individual) {
        List<Double> weights = individual.network.getFlattenedWeights();
        for (int i = 0; i < weights.size(); i++) {
            if (random.nextDouble() < p) {
                double mutation = random.nextGaussian() * K;
                weights.set(i, weights.get(i) + mutation);
            }
        }
        individual.network.setWeightsFromFlattened(weights);
    }

    private static class Individual {
        NeuralNetwork network;
        double fitness;

        Individual(NeuralNetwork network) {
            this.network = network;
            this.fitness = 0;
        }
    }
}