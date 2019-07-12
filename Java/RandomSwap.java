/*
 * Random Swap implementation for Clustering Methods course
 * Uses jblas library http://jblas.org/
 * 
 * Check running.txt for instructions for compiling and running this program.
 * 
 * Hannu Sillanpää
 * 
 */

import org.jblas.*;
import static org.jblas.DoubleMatrix.*;
import static org.jblas.MatrixFunctions.*;
import static org.jblas.util.Permutations.*;
import org.jblas.ranges.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.PrintWriter;
import java.util.Random;
import java.lang.Math.*;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Scanner;

public class RandomSwap {
	
	private enum Algorithm {
		KMEANS, FAST_KMEANS, RS_KM, RS_FKM
	}
	
	private static int TEST_COUNT = 100;// number of clusterings made when running tests
	private static int CLUSTER_TEST_COUNT = 100;// number of tests for cluster count test
	private static int SWAP_COUNT = 500;// number of attempted swaps in the random swap algorithm
	private static Algorithm ALGORITHM = Algorithm.RS_FKM;
	
	private static Random rng = new Random();
	
	// Saves a DoubleMatrix object in plain text format
	private static void saveMatrixAsText(DoubleMatrix data, String fileName, boolean cut)
			throws IOException {
		Path file = Paths.get(fileName);
		Files.deleteIfExists(file);
		PrintWriter writer = new PrintWriter(fileName, "UTF-8");
		for (DoubleMatrix row : data.rowsAsList()) {
			if (cut) {
				writer.print((int)row.get(0));
			} else {
				writer.print(row.get(0));
			}
			for (int i = 1; i < row.length; i++) {
				if (cut) {
					writer.print(" " + (int)row.get(i));
				} else {
					writer.print(" " + row.get(i));
				}
			}
			writer.println();
		}
		writer.close();
	}
	
	// brute force k nearest neighbours
	private static DoubleMatrix knn(DoubleMatrix data, int k) {
		DoubleMatrix result = zeros(data.rows, k);// knn graph as table
		for (int v = 0; v < data.rows; v++) {
			DoubleMatrix distances = pow(data.subRowVector(data.getRow(v)), 2).rowSums();
			int[] nearest = Arrays.copyOfRange(distances.sortingPermutation(), 1, k+1);
			// DoubleMatrix constructor needs double[], so nearest needs to be converted
			// another option would be to use int[][] for knn graph instead of DoubleMatrix
			double[] nearestD = new double[k];
			for (int i = 0; i < k; i++) {
				nearestD[i] = nearest[i];
			}
			result.putRow(v, new DoubleMatrix(nearestD));
		}
		return result;
	}
	
	// k-means partition, modifies labels! uses a knn graph of centroids
	// Method was used for some exercises related to knn and clustering
	private static DoubleMatrix partition_knn(DoubleMatrix data, DoubleMatrix labels, DoubleMatrix centroids) {
		DoubleMatrix knnGraph = knn(centroids, 3);
		for (int i = 0; i < data.rows; i++) {
			// check labelled (=previous) and k nearest centroid distances
			int label = (int)labels.get(i);
			double labelDist = pow(centroids.getRow(label).sub(data.getRow(i)), 2).sum();
			DoubleMatrix graphNode = knnGraph.getRow(label);
			DoubleMatrix nearestCentroids = centroids.getRows(new IndicesRange(graphNode));
			DoubleMatrix distances = pow(nearestCentroids.subRowVector(data.getRow(i)), 2).rowSums();
			int nearest = distances.argmin();
			if (distances.get(nearest) < labelDist) {
				labels.put(i, graphNode.get(nearest));
			}
		}
		return labels;
	}
	
	// k-means partition, modifies labels!
	private static DoubleMatrix partition(DoubleMatrix data, DoubleMatrix labels, DoubleMatrix centroids) {
		DoubleMatrix distances;
		for (int v = 0; v < data.rows; v++) {
			distances = powi(centroids.subRowVector(data.getRow(v)), 2).rowSums();
			labels.put(v, distances.argmin());
		}
		return labels;
	}
	
	// Centroid calculation for k-means
	private static void calculateCentroids(DoubleMatrix data, DoubleMatrix clusterLabels, DoubleMatrix centroids,
			DoubleMatrix prevCentroids) {
		int nData = data.rows;
		int nDims = data.columns;
		int nClusters = centroids.rows;
		DoubleMatrix labelCounts = zeros(nClusters);
		
		Arrays.fill(centroids.data, 0);
		// Assemble sums of vectors belonging to the various partitions into the centroids matrix
		for (int v = 0; v < nData; v++) {
			int label = (int)clusterLabels.get(v);
			// Count the labels
			labelCounts.data[label]++;
			// Add datapoint to the sum of it's partition in the centroids matrix
			// couldn't find method for adding to a specific row of DoubleMatrix, so have to do it myself...
			int dataStartIndex = v;
			int centroidsIndex = label;// index in centroid matrix data
			for (int i = dataStartIndex; i < dataStartIndex+nDims*nData; i+=nData) {
				centroids.data[centroidsIndex] += data.data[i];
				centroidsIndex += nClusters;
			}
		}
		// Divide the sums of data vectors in the centroid matrix by the label counts to obtain
		// the means of the partitions
		centroids.diviColumnVector(labelCounts);
		// check for empty clusters and use previous centroid values for them
		for (int v = 0; v < nClusters; v++) {
			if (labelCounts.get(v) == 0) {
				centroids.putRow(v, prevCentroids.getRow(v));
				//System.out.println("Empty cluster");
			}
		}
	}
	
	// runs k-means with default initialization and parameters
	private static DoubleMatrix[] kmeans(DoubleMatrix data, int nClusters) {
		return kmeans(data, nClusters, 20000);
	}
	
	// k-means including random centroid initialization
	private static DoubleMatrix[] kmeans(DoubleMatrix data, int nClusters, int maxIterations) {
		int nData = data.rows;
		DoubleMatrix centroids;
		
		// init clusters
		int[] rands = randomSubset(nClusters, nData);// indexes for initial centroids
		centroids = data.getRows(rands);
		
		return kmeans(data, nClusters, centroids, maxIterations);// todo dont leave this
	}
	
	// Regular k-means algorithm
	// note: centroids is modified
	private static DoubleMatrix[] kmeans(DoubleMatrix data, int nClusters, DoubleMatrix centroids,
			int maxIterations) {// nClusters not necessarily needed here
		int nData = data.rows;
		DoubleMatrix clusterLabels = zeros(nData);
		
		int it = 0;
		DoubleMatrix prevCentroids = centroids.dup();
		do {
			it++;
			//System.out.println("-------- " + it + " ----------");
			partition(data, clusterLabels, centroids);
			prevCentroids.copy(centroids);
			calculateCentroids(data, clusterLabels, centroids, prevCentroids);
			
			if (it > maxIterations) break;
			if (it > 10000) {
				System.out.println("it over 10000 !!!!");
				break;
			}
			//System.out.println(change);
		} while (!prevCentroids.equals(centroids));
		//System.out.println(it);
		
		DoubleMatrix[] results = {clusterLabels, centroids};
		return results;
	}
	
	// Squared distance between two row vectors picked from matrices
	// Proportions are assumed to be correct!
	// A lot faster than powi(mat1.getRow(row1).subiRowVector(mat2.getRow(row2)), 2).sum();
	private static double calculateDistance(DoubleMatrix mat1, int row1, DoubleMatrix mat2, int row2) {
		double result = 0;
		int mat1Rows = mat1.rows;
		int mat2Rows = mat2.rows;
		int nDims = mat1.columns;
		int index2 = row2;
		double sub;
		for (int index1 = row1; index1 < nDims*mat1Rows; index1 += mat1Rows) {
			sub = mat1.data[index1] - mat2.data[index2];
			result += sub * sub;
			index2 += mat2Rows;
		}
		return result;
	}
	
	// runs fast k-means with default initialization and parameters
	private static DoubleMatrix[] fast_kmeans(DoubleMatrix data, int nClusters) {
		return fast_kmeans(data, nClusters, 20000);
	}
	
	// fast k-means including random centroid initialization
	private static DoubleMatrix[] fast_kmeans(DoubleMatrix data, int nClusters, int maxIterations) {
		int nData = data.rows;
		DoubleMatrix centroids;
		
		// init clusters
		int[] rands = randomSubset(nClusters, nData);// indexes for initial centroids
		centroids = data.getRows(rands);
		
		return fast_kmeans(data, nClusters, centroids, maxIterations);
	}
	
	// fast k-means with partition initialization
	private static DoubleMatrix[] fast_kmeans(DoubleMatrix data, int nClusters, DoubleMatrix centroids,
			int maxIterations) {
		DoubleMatrix clusterLabels = zeros(data.rows);
		partition(data, clusterLabels, centroids);
		return fast_kmeans(data, nClusters, centroids, clusterLabels, maxIterations);
	}
	
	// fast k-means without initialization
	private static DoubleMatrix[] fast_kmeans(DoubleMatrix data, int nClusters, DoubleMatrix centroids,
			DoubleMatrix clusterLabels, int maxIterations) {
		int nData = data.rows;
		
		int it = 0;
		DoubleMatrix distances;
		LinkedList<Integer> activeCentroids = new LinkedList<Integer>();
		boolean[] centroidActiveness = new boolean[nClusters];
		
		DoubleMatrix prevCentroids = centroids.dup();
		while (true) {
			it++;
			prevCentroids.copy(centroids);
			// Calculate centroids
			calculateCentroids(data, clusterLabels, centroids, prevCentroids);
			// Detect changed centroids
			activeCentroids.clear();
			for (int v = 0; v < nClusters; v++) {
				centroidActiveness[v] = false;
				if (!centroids.getRow(v).equals(prevCentroids.getRow(v))) {
					activeCentroids.add(v);
					centroidActiveness[v] = true;
				}
			}
			//System.out.println(activeCentroids.size());
			// Check for convergence
			if (activeCentroids.isEmpty()) {
				break;
			}
			// Reduced search partition
			for (int v = 0; v < nData; v++) {
				int label = (int)clusterLabels.get(v);
				if (centroidActiveness[label]) {
					// full search
					distances = powi(centroids.subRowVector(data.getRow(v)), 2).rowSums();
					clusterLabels.put(v, distances.argmin());
				} else {
					// partial search from active vectors
					//double minDistance = powi(centroids.getRow(label).subiRowVector(data.getRow(v)), 2).sum();
					double minDistance = calculateDistance(centroids, label, data, v);
					for (Integer i : activeCentroids) {
						//double distance = powi(centroids.getRow(i).subiRowVector(data.getRow(v)), 2).sum();
						double distance = calculateDistance(centroids, i, data, v);
						if (distance < minDistance) {
							minDistance = distance;
							label = i;
						}
					}
					clusterLabels.put(v, label);
				}
			}
			if (it > maxIterations) break;
			if (it > 10000) {
				System.out.println("it over 10000 !!!!");
				break;
			}
		}
		//System.out.println(it);
		
		DoubleMatrix[] results = {clusterLabels, centroids};
		return results;
	}
	
	// Total squared error calculation for clustering results
	private static double tse(DoubleMatrix data, DoubleMatrix clusterLabels, DoubleMatrix centroids) {
		int nClusters = centroids.rows;
		double error = 0;
		for (int v = 0; v < nClusters; v++) {
			DoubleMatrix clusterData = data.getRows(clusterLabels.eq(v));
			error += powi(clusterData.subiRowVector(centroids.getRow(v)), 2).sum();
		}
		return error;
	}
	
	// variance between clusters
	private static double ssb(DoubleMatrix data, DoubleMatrix clusterLabels, DoubleMatrix centroids) {
		DoubleMatrix dataAverage = data.columnMeans();
		double result = 0;
		for (int i = 0; i < centroids.rows; i++) {
			result += clusterLabels.eq(i).sum() * powi(centroids.getRow(i).sub(dataAverage), 2).sum();
		}
		return result;
	}
	
	// Random swap using fast k-means and local repartition
	private static DoubleMatrix[] randomSwap_fkm(DoubleMatrix data, int nClusters, int nSwaps) {
		int nData = data.rows;
		DoubleMatrix centroids;
		
		// init clusters
		int[] rands = randomSubset(nClusters, nData);// indexes for initial centroids
		centroids = data.getRows(rands);
		
		// initial partitioning
		DoubleMatrix clusterLabels = zeros(nData);
		partition(data, clusterLabels, centroids);
		
		DoubleMatrix[] currentResults = {clusterLabels, centroids};
		double currentError = tse(data, currentResults[0], currentResults[1]);
		
		int successes = 0;
		int failures = 0;
		DoubleMatrix distances;
		
		for (int i = 0; i < nSwaps; i++) {
			DoubleMatrix newCentroids = currentResults[1].dup();
			DoubleMatrix newPartition = currentResults[0].dup();
			// Random swap
			int swappedCluster = rng.nextInt(nClusters);
			newCentroids.putRow(swappedCluster, data.getRow(rng.nextInt(nData)));
			// Local repartition
			for (int j = 0; j < nData; j++) {
				if (newPartition.get(j) == swappedCluster) {
					// Re-allocate data from old cluster
					distances = powi(newCentroids.subRowVector(data.getRow(j)), 2).rowSums();
					newPartition.put(j, distances.argmin());
				} else {
					// Find data for swapped cluster: compare currently assigned centroid and new centroid for rest of data
					double distanceToCurrent = calculateDistance(newCentroids, (int)newPartition.get(j), data, j);
					double distanceToSwapped = calculateDistance(newCentroids, swappedCluster, data, j);
					if (distanceToSwapped < distanceToCurrent) {
						newPartition.put(j, swappedCluster);
					}
				}
			}
			DoubleMatrix[] newResults = fast_kmeans(data, nClusters, newCentroids, newPartition, 5);
			double newError = tse(data, newResults[0], newResults[1]);
			// Compare error values and roll back if no improvement
			if (newError < currentError) {
				currentResults = newResults;
				currentError = newError;
				successes++;
				//System.out.print("X");
			} else {
				failures++;
				//System.out.print("-");
			}
		}
		//System.out.println();
		
		//System.out.println(successes + " successes, " + failures + " failures");
		
		return currentResults;
	}
	
	// random swap without fast k-means and local repartition
	private static DoubleMatrix[] randomSwap(DoubleMatrix data, int nClusters, int nSwaps) {
		int nData = data.rows;
		DoubleMatrix centroids;
		
		// init clusters
		int[] rands = randomSubset(nClusters, nData);// indexes for initial centroids
		centroids = data.getRows(rands);
		
		// no need to do partitioning here, kmeans does it anyway
		
		DoubleMatrix[] currentResults = kmeans(data, nClusters, centroids, 5);
		double currentError = tse(data, currentResults[0], currentResults[1]);
		
		int successes = 0;
		int failures = 0;
		
		for (int i = 0; i < nSwaps; i++) {
			DoubleMatrix newCentroids = currentResults[1].dup();
			// random swap
			newCentroids.putRow(rng.nextInt(nClusters), data.getRow(rng.nextInt(nData)));
			DoubleMatrix[] newResults = kmeans(data, nClusters, newCentroids, 5);
			double newError = tse(data, newResults[0], newResults[1]);
			if (newError < currentError) {
				currentResults = newResults;
				currentError = newError;
				successes++;
				//System.out.print("X");
			} else {
				failures++;
				//System.out.print("-");
			}
		}
		//System.out.println();
		
		//System.out.println(successes + " successes, " + failures + " failures");
		
		return currentResults;
	}
	
	// subfunction for finding orphans in a single mapping direction
	// used by the centroidIndex method
	private static int ciSub(DoubleMatrix aCentroids, DoubleMatrix bCentroids) {
		// find nearest centroids, mark clusters that are found
		int nClusters = aCentroids.rows;
		DoubleMatrix aNearest = zeros(nClusters);
		DoubleMatrix bOrphans = ones(nClusters);
		for (int i = 0; i < nClusters; i++) {
			DoubleMatrix distances = powi(bCentroids.subRowVector(aCentroids.getRow(i)), 2).rowSums();
			int closest = distances.argmin();
			aNearest.put(i, closest);
			bOrphans.put(closest, 0);// centroid no. i in B is no longer an orphan
		}
		// now the amount of orphans can be found by taking a sum of bOrphans
		return (int)bOrphans.sum();
	}
	
	// Calculates centroid index
	private static int centroidIndex(DoubleMatrix aCentroids, DoubleMatrix bCentroids) {
		aCentroids.assertSameSize(bCentroids);
		return Math.max(ciSub(aCentroids, bCentroids), ciSub(bCentroids, aCentroids));
	}
	
	private static DoubleMatrix[] runChosenAlgorithm(DoubleMatrix data, int nClusters) {
		DoubleMatrix[] results;
		switch (ALGORITHM) {
			case KMEANS:
				results = kmeans(data, nClusters);
				break;
			case FAST_KMEANS:
				results = fast_kmeans(data, nClusters);
				break;
			case RS_KM:
				results = randomSwap_fkm(data, nClusters, SWAP_COUNT);
				break;
			default:
				results = randomSwap_fkm(data, nClusters, SWAP_COUNT);
				break;
		}
		return results;
	}
	
	// Runs a clustering algorithm a number of times and reports measurements from the results.
	// Mean SSE, Min SSE, Mean nMSE, Mean CI, Success%, Mean epsilon,
	// Average time and total times
	private static void runTests(DoubleMatrix data, DoubleMatrix realCentroids, int nClusters) {
		long testStartTime = System.currentTimeMillis();
		
		int nData = data.rows;
		int nDims = data.columns;
		DoubleMatrix realLabels;
		double bestTse = 0;
		if (realCentroids != null) {
			nClusters = realCentroids.rows;
			realLabels = partition(data, zeros(nData), realCentroids);
			bestTse = tse(data, realLabels, realCentroids);
		}
		DoubleMatrix tseValues = zeros(TEST_COUNT);
		DoubleMatrix ciValues = zeros(TEST_COUNT);
		DoubleMatrix times = zeros(TEST_COUNT);
		
		for (int i = 0; i < TEST_COUNT; i++) {
			System.out.print(i + 1 + " ");
			long startTime = System.currentTimeMillis();
			DoubleMatrix[] results = runChosenAlgorithm(data, nClusters);
			double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0f;
			times.put(i, elapsedTime);
			//System.out.println("" + elapsedTime);
			tseValues.put(i, tse(data, results[0], results[1]));
			if (realCentroids != null) {
				ciValues.put(i, centroidIndex(results[1], realCentroids));
			}
		}
		System.out.println();
		
		double testElapsedTime = (System.currentTimeMillis() - testStartTime) / 1000.0f;
		
		double meanTse = tseValues.mean();
		System.out.println("##########");
		System.out.println("Results: (" + SWAP_COUNT + " swaps, " + TEST_COUNT + " tests)");
		System.out.println("Ground truth SSE: " + bestTse);
		System.out.println("Mean SSE: " + meanTse);
		System.out.println("Min SSE: " + tseValues.min());
		System.out.println("nMSE: " + meanTse / (nData * nDims));
		if (realCentroids != null) {
			System.out.println("CI: " + ciValues.mean());
			System.out.println("Success: " + ciValues.eq(0).sum() / TEST_COUNT * 100 + " %");
			System.out.println("Epsilon: " + (meanTse - bestTse) / bestTse);
		}
		System.out.println("Average time: " + times.mean() + " s");
		System.out.println("Sum of times: " + times.sum() + " s");
		System.out.println("Total test time: " + testElapsedTime + " s");
	}
	
	// cluster count tests for ex7
	private static void clusterCountTests(DoubleMatrix data, int maxCount) {
		System.out.println("Cluster count test: (min of " + CLUSTER_TEST_COUNT + " tests) (tse, wb-ratio)");
		for (int c = 1; c <= maxCount; c++) {
			double minTse = Double.MAX_VALUE;
			double minRatio = Double.MAX_VALUE;
			for (int i = 0; i < CLUSTER_TEST_COUNT; i++) {
				DoubleMatrix[] results = runChosenAlgorithm(data, c);
				double tseValue = tse(data, results[0], results[1]);
				double ratio = tseValue / ssb(data, results[0], results[1]);
				if (tseValue < minTse) {
					minTse = tseValue;
				}
				if (ratio < minRatio) {
					minRatio = ratio;
				}
			}
			System.out.println("" + minTse + "\t\t" + minRatio);
		}
		System.out.println("done");
	}
	
	private static void printHelp() {
		System.out.println("Check help.txt for instructions.");
	}
	
	private static void printParseError(String location) {
		System.out.println("Argument parsing error in " + location);
	}
	
	private static void printClusteringAlgorithm() {
		switch (ALGORITHM) {
			case KMEANS:
				System.out.println("k-means");
				break;
			case FAST_KMEANS:
				System.out.println("fast k-means");
				break;
			case RS_KM:
				System.out.println("random swap with regular k-means");
				break;
			default:
				System.out.println("random swap with fast k-means");
				break;
		}
	}
	
	public static void main(String args[]) throws IOException {
		String groundTruthFile = null;
		boolean pause = false;
		boolean runTests = false;
		boolean runClusterCountTests = false;
		// Parse command line switches
		int i;
		for (i = 0; i < args.length - 2; i++) {
			if (args[i].equals("-a")) {
				if (i > args.length - 3) {
					printParseError("-a");
					return;
				}
				if (args[i+1].equals("km")) {
					ALGORITHM = Algorithm.KMEANS;
				} else if (args[i+1].equals("fkm")) {
					ALGORITHM = Algorithm.FAST_KMEANS;
				} else if (args[i+1].equals("rs-km")) {
					ALGORITHM = Algorithm.RS_KM;
				} else if (args[i+1].equals("rs-fkm")) {
					ALGORITHM = Algorithm.RS_FKM;
				} else {
					System.out.println("Unknown algorithm " + args[i+1]);
					printHelp();
				}
				i++;
			} else if (args[i].equals("-swaps")) {
				if (i > args.length - 3) {
					printParseError("-swaps");
					return;
				}
				try {
					SWAP_COUNT = Integer.valueOf(args[i+1]);
					System.out.println("Swaps set to " + SWAP_COUNT);
				} catch (NumberFormatException e) {
					printParseError("-swaps");
					return;
				}
				i++;
			} else if (args[i].equals("-tests")) {
				if (i > args.length - 3) {
					printParseError("-tests");
					return;
				}
				try {
					TEST_COUNT = Integer.valueOf(args[i+1]);
				} catch (NumberFormatException e) {
					printParseError("-tests");
					return;
				}
				runTests = true;
				i++;
			} else if (args[i].equals("-gt")) {
				if (i > args.length - 3) {
					printParseError("-gt");
					return;
				}
				groundTruthFile = args[i+1];
				if (groundTruthFile.endsWith(".txt")) {
					groundTruthFile = groundTruthFile.substring(0, groundTruthFile.length() - 4);
				}
				if (!Files.isRegularFile(Paths.get(groundTruthFile + ".txt"))) {
					System.out.println("Ground truth file " + groundTruthFile + ".txt not found.");
					return;
				}
				System.out.println("Using ground truth file " + groundTruthFile + ".txt");
				i++;
			} else if (args[i].equals("-c")) {
				if (i > args.length - 3) {
					printParseError("-c");
					return;
				}
				try {
					CLUSTER_TEST_COUNT = Integer.valueOf(args[i+1]);
				} catch (NumberFormatException e) {
					printParseError("-c");
					return;
				}
				runClusterCountTests = true;
				i++;
			} else if (args[i].equals("-h") || args[i].equals("--help")) {
				printHelp();
				return;
			} else if (args[i].equals("-p")) {
				pause = true;
			} else {
				System.out.println("Unknown argument " + args[i]);
				printHelp();
				return;
			}
		}
		// Parse data file name and cluster count
		if (args.length - i < 2) {
			System.out.println("Not enough arguments");
			printHelp();
			return;
		}
		String fileName = args[args.length - 2];
		if (fileName.endsWith(".txt")) {
			fileName = fileName.substring(0, fileName.length() - 4);
		}
		if (!Files.isRegularFile(Paths.get(fileName + ".txt"))) {
			System.out.println("Data file " + fileName + ".txt not found.");
			return;
		}
		int nClusters;
		try {
			nClusters = Integer.valueOf(args[args.length - 1]);
		} catch (NumberFormatException e) {
			printParseError("cluster count");
			return;
		}
		
		if (pause) {
			Scanner scan = new Scanner(System.in);
			System.out.println("Press enter");
			scan.nextLine();
		}
		
		System.out.println("Loading data file");
		DoubleMatrix data = loadAsciiFile(fileName + ".txt");
		
		if (groundTruthFile == null) {
			groundTruthFile = fileName + "-cb.txt";
		}
		
		if (runTests) {
			System.out.print("Running " + TEST_COUNT + " tests using ");
			printClusteringAlgorithm();
			DoubleMatrix realCentroids = null;
			if (Files.isRegularFile(Paths.get(groundTruthFile))) {
				realCentroids = loadAsciiFile(fileName + "-cb.txt");
			}
			runTests(data, realCentroids, nClusters);
		}
		if (runClusterCountTests) {
			System.out.print("Using ");
			printClusteringAlgorithm();
			clusterCountTests(data, nClusters);
		}
		if (!runTests && !runClusterCountTests) {
			System.out.print("Running ");
			printClusteringAlgorithm();
			DoubleMatrix[] results;
			switch (ALGORITHM) {
				case KMEANS:
					results = kmeans(data, nClusters);
					break;
				case FAST_KMEANS:
					results = fast_kmeans(data, nClusters);
					break;
				case RS_KM:
					results = randomSwap(data, nClusters, SWAP_COUNT);
					break;
				default:
					results = randomSwap_fkm(data, nClusters, SWAP_COUNT);
					break;
			}
			System.out.println("TSE: " + tse(data, results[0], results[1]));
			if (Files.isRegularFile(Paths.get(groundTruthFile))) {
				DoubleMatrix realCentroids = loadAsciiFile(groundTruthFile);
				System.out.println("CI: " + centroidIndex(results[1], realCentroids));
			}
			System.out.println("saving results");
			saveMatrixAsText(results[0].add(1), "results_partition.txt", true);// +1 for clusterator?
			saveMatrixAsText(results[1], "results_centroids.txt", false);
			System.out.println("Done");
		}
	}
}
