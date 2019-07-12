Compiling on Windows:
javac -cp jblas-1.2.4.jar;. RandomSwap.java
Running on Windows:
java -cp jblas-1.2.4.jar;. RandomSwap [arguments] <datafile> <cluster count>

Compiling on Linux:
javac -cp jblas-1.2.4.jar:. RandomSwap.java
Running on Linux:
java -cp jblas-1.2.4.jar:. RandomSwap [arguments] <datafile> <cluster count>


The data files are expected to be in .txt format.
The ground truth centroids are expected to be in a text file called <original>-cb.txt,
unless using the -gt argument.
Example data file name: "s1.txt"
Example ground truth file name: "s1-cb.txt"



Command line arguments:

-a km, -a fkm, -a rs-km, -a rs-fkm: Clustering algorithm selection, default rs-fkm
k-means, fast k-means, random swap w/ k-means, random swap w/ fast k-means

-swaps X: Swap count for random swap, defaults to 500

-tests X: Run validation tests, including SSE and CI, X is the number of clusterings generated for the test
filename-cb.txt will be used for ground truth centroids if found, the file can also be specified by -gt

-gt filename.txt: Specify ground truth centroid file for validation tests, defaults to filename-cb.txt

-c X: Run cluster count tests: print min-TSE and min-WB-ratio from X tests with k going from 1 to <cluster count>

-h, --help: Show command line help

-p: pause before execution (used for debugging)

Without using the -tests or -c arguments the program will run the clustering algorithm once
and save the results to results_partition.txt and results_centroids.txt.

Syntax:
java -cp jblas-1.2.4.jar;. RandomSwap [arguments] <datafile> <cluster count>

Examples:
Run tests for dataset s1.txt 100 times, using default algorithm (Random swap with fast k-means)
java -cp jblas-1.2.4.jar;. RandomSwap -tests 100 s1 15

Run tests for dataset s1.txt 100 times, using regular k-means
java -cp jblas-1.2.4.jar;. RandomSwap -tests 100 -a km s1 15

