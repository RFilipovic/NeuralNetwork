package ui;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class Solution {

	private static String trainFile = "";
	private static String testFile = "";
	private static String architecture = "";
	private static int popsize = 0;
	private static int elitism = 0;
	private static double p = 0;
	private static double K = 0;
	private static int iter = 0;

	public static void main(String... args) {
		setArgs(args);

		Map<String, Object> trainData = parseFile(trainFile);
		Map<String, Object> testData = parseFile(testFile);

		List<String> headers = (List<String>) trainData.get("headers");
		List<Double[]> trainSet = (List<Double[]>) trainData.get("data");
		List<Double[]> testSet = (List<Double[]>) testData.get("data");

		int inputDim = headers.size() - 1;
		GeneticAlg ga = new GeneticAlg(inputDim, architecture, popsize, elitism, p, K, iter);
		NeuralNetwork bestNetwork = ga.evolve(trainSet);

		double testError = bestNetwork.calculateMSE(testSet);
		System.out.printf("[Test error]: %.6f\n", testError);
	}

	private static void setArgs(String... args) {
		for (int i = 0; i < args.length; i++){
			switch (args[i]) {
				case "--test" -> testFile = args[i + 1];
				case "--train" -> trainFile = args[i + 1];
				case "--nn" -> architecture = args[i + 1];
				case "--popsize" -> popsize = Integer.parseInt(args[i + 1]);
				case "--elitism" -> elitism = Integer.parseInt(args[i + 1]);
				case "--p" -> p = Double.parseDouble(args[i + 1]);
				case "--K" -> K = Double.parseDouble(args[i + 1]);
				case "--iter" -> iter = Integer.parseInt(args[i + 1]);
			}
		}
	}

	private static Map<String, Object> parseFile(String fileName) {
		File file = new File(fileName);
		List<Double[]> data = new ArrayList<>();
		List<String> headers = new ArrayList<>();

		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
			String line;
			if ((line = br.readLine()) != null) {
				String[] headerParts = line.split(",");
				headers.addAll(Arrays.asList(headerParts));
			}

			while ((line = br.readLine()) != null) {
				String[] parts = line.split(",");
				Double[] row = new Double[parts.length];
				for(int i = 0; i < parts.length; i++){
					row[i] = Double.parseDouble(parts[i]);
				}
				data.add(row);
			}
		} catch (IOException e) {
			throw new RuntimeException("Error reading file: " + fileName, e);
		}

		Map<String, Object> result = new HashMap<>();
		result.put("headers", headers);
		result.put("data", data);
		return result;
	}
}