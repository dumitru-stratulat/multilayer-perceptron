package MLP;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.Scanner;
import java.util.Arrays;

public class MLP {
	/* Input files path */
	private static final String FIRST_DATASET = "./resources/cw2DataSet1.csv";
	private static final String SECOND_DATASET = "./resources/cw2DataSet2.csv";
	
	/* Weights files constants*/
	private static final String HIDDEN_LAYER_WEIGHTS = "./resources/hiddenLayerWeights.txt";
	private static final String OUTPUT_LAYER_WEIGHTS = "./resources/outputLayerWeights.txt";
	

	private static final int TOTAL_COLUMNS = 65;
	private static final int TOTAL_ROWS = 2810;
	private static final int TOTAL_INPUT_LAYER_NEURONS = 64;
	private static final int TOTAL_HIDDEN_LAYER_NEURONS = 50;
	private static final int TOTAL_OUTPUT_LAYER_NEURONS = 10;

	
	/* Weights */
	private static double[][] inputLayerToHiddenLayerWeights = new double[TOTAL_INPUT_LAYER_NEURONS][TOTAL_HIDDEN_LAYER_NEURONS];
	private static double[][] hiddenLayerToOutputLayerWeights = new double[TOTAL_HIDDEN_LAYER_NEURONS][TOTAL_OUTPUT_LAYER_NEURONS];
	
	/* Layers neurons */
	private static double[][] hiddenLayerNeuron = new double[TOTAL_ROWS][TOTAL_HIDDEN_LAYER_NEURONS];
	private static double[][] outputLayerNeuron = new double[TOTAL_ROWS][TOTAL_OUTPUT_LAYER_NEURONS];
	
	/* Arrays of inputss from files */
	private static double[][] firstFileDataArray = new double[TOTAL_ROWS][TOTAL_COLUMNS];
	private static double[][] secondFileDataArray = new double[TOTAL_ROWS][TOTAL_COLUMNS];
	

	 /*
	 	Method read inputs of MLP from file
   	 	Reading every line in CSV file and returns 2d array of inputs for MLP
	 */


	private static double[][] readingFromFile(String fileName, double[][] fileDataArray) {
		Scanner scanner;
		String inputLine;
		int rowCounter = 0;
		
		try {
			scanner = new Scanner(new BufferedReader(new FileReader(fileName)));
			
			while(scanner.hasNextLine()) {
				inputLine = scanner.nextLine();
				String[] commaDelimiter = inputLine.split(",");
				// Split elements separated by comma from the file
				String[] inputArray = commaDelimiter;
				
				for(int arrayElement = 0; arrayElement < inputArray.length; arrayElement++) {
					fileDataArray[rowCounter][arrayElement] = Double.parseDouble(inputArray[arrayElement]);
				}
				// Count totalrows
				rowCounter++;
			}
		// Error handler
		} catch (FileNotFoundException errorName) {
			System.out.println("File not found: " + errorName);
		} catch (IndexOutOfBoundsException errorName) {
			System.out.println("Error reading file: " + errorName);
		}
		return fileDataArray;
	}
	/*
		Method saves data in file
	*/
	public static void writeToFile(String fileName, double[][] data) {
		// Create PrintStream class object
        PrintStream printStreamObj;
        double arrayElement;
        String path = "resources/";
	    try {
	    	printStreamObj = new PrintStream(new FileOutputStream(path + fileName));
	        for(int row = 0; row < data.length; row++){
	        	if(row != 0)
	        		// Next line
	        		printStreamObj.println();
	           for(int column = 0; column < data[row].length; column++){
	                    arrayElement = data[row][column];
	                    // Separate by comma;
	                    printStreamObj.print(arrayElement + ",");
			   }
			}
	        // Close the writing into file
	        printStreamObj.close();
	        } catch (FileNotFoundException errorName) {
	            System.out.println("File was not created. ERROR: " + errorName.getMessage());
	        }
	}
	

	// Method read data from file and save in array

	public static void storeData() {
		readingFromFile(FIRST_DATASET, firstFileDataArray);
		readingFromFile(SECOND_DATASET, secondFileDataArray);
		readingFromFile(HIDDEN_LAYER_WEIGHTS, inputLayerToHiddenLayerWeights);
		readingFromFile(OUTPUT_LAYER_WEIGHTS, hiddenLayerToOutputLayerWeights);
	}
	// Sigmoid derivative formula function.
	private static double sigmoidFunctionDerivative(double value) {
		return (value * (1 - value));
	}
	
	// Sigmoid function formula.
    private static double sigmoid(double value) {
        return (1 / (1 + Math.exp(-value)));
    }

	/*
		Method that calculates the dot product of two layers
		As parameters takes currentLayerNeurons, nextLayerNeuronss and weightValue
	 */
	public static double[][] dotProduct(double[][] currentLayer, double[][] nextLayer, double[][] weightsValue){
    	double neuronValue = 0;
        // Loop through current layer neurons
        for(int startingNeuron = 0; startingNeuron < currentLayer.length; startingNeuron++) {
	        // Loop through next layer neurons
	        for(int nextLayerNeuron = 0; nextLayerNeuron < nextLayer[startingNeuron].length; nextLayerNeuron++) {
	            // Loop through input layer neurons and reset neuronValue before each loop
	            neuronValue = 0;
	                for(int currentLayerNeuron = 0; currentLayerNeuron < currentLayer[startingNeuron].length - 1; currentLayerNeuron++){
	                	// Count total neuronValue for one next layer neuron
	                    neuronValue += currentLayer[startingNeuron][currentLayerNeuron] * weightsValue[currentLayerNeuron][nextLayerNeuron];
	                }
	                nextLayer[startingNeuron][nextLayerNeuron] = sigmoid(neuronValue);;
	        }
        }
	    return nextLayer;
    }

	//Calculates dot products of 2 layers
    public static void calculateDotProducts(double[][] inputLayerArray) {
    	// Calculate dot product from input to hidden layer
    	dotProduct(inputLayerArray, hiddenLayerNeuron, inputLayerToHiddenLayerWeights);
	
    	// Calculate dot product from hidden to output layer
    	dotProduct(hiddenLayerNeuron, outputLayerNeuron, hiddenLayerToOutputLayerWeights);
    }
    
    /* Method that trains the NN using backpropagation method
	 * Takes 5 parameters:
	 * 1st parameter input values
	 * 2nd parameter is learningRate:
	 * 3rd parameter is a leastMeanSquaredError (least mean squared error):
	 * 4th parameter is momentum
	 * 5th parameter is  maxEpochs it iss iterations that train will run
	 */
    private static void train(double[][] input, double learningRate, double leastMeanSquaredError, double momentum, int maxEpochs) {
    	// Mean squared error variable
    	double meanSquaredError = 0.0;
    	
    	// Epochs counter variable
    	int epochCounter = 1;
    	
    	// Total totalError variable
    	double totalError = 0.0;
    	
    	// Lowest totalError threshold variable
    	double errorThreshold = 0.0001;
    	
    	// Target variable (last element of input array)
    	double target;
    	
    	// Target position in array
    	int targetPosition = TOTAL_COLUMNS - 1;
    	
    	// Hidden layer delta
    	double[][] hiddenLayerDelta = new double[TOTAL_ROWS][TOTAL_HIDDEN_LAYER_NEURONS];
    	
    	// Output layer delta
    	double[][] outputLayerDelta = new double[TOTAL_ROWS][TOTAL_OUTPUT_LAYER_NEURONS];

    	// Temporary weights for training
    	double[][] temporaryHiddenWeights = Arrays.copyOf(inputLayerToHiddenLayerWeights, inputLayerToHiddenLayerWeights.length);
    	double[][] temporaryOutputWeights = Arrays.copyOf(hiddenLayerToOutputLayerWeights, hiddenLayerToOutputLayerWeights.length);

    	// Previous weights for training
       	double[][] previousHiddenWeights = Arrays.copyOf(inputLayerToHiddenLayerWeights, inputLayerToHiddenLayerWeights.length);
    	double[][] previousOutputWeights = Arrays.copyOf(hiddenLayerToOutputLayerWeights, hiddenLayerToOutputLayerWeights.length);

        // Loop until errorThreshold is reached
        while(Math.abs(meanSquaredError - leastMeanSquaredError) > errorThreshold) {
            // For each epoch reset the mean square totalError
            meanSquaredError = 0.0;
        	
            // Loop through all the inputs
            for(int fileArray = 0; fileArray < input.length; fileArray++) {
            
            	// Calculate dot products from input to hidden and from hidden to output
            	calculateDotProducts(input);
            	
        		// Set the target which is the last element of input array
            	target = input[fileArray][targetPosition];
  
                // Backpropagation from output layer
                for(int outputNeuron = 0; outputNeuron < outputLayerNeuron[fileArray].length; outputNeuron++) {
                    // Calculate delta and totalError if output neuron IS NOT the target
                    if(outputNeuron != target) {
                        outputLayerDelta[fileArray][outputNeuron] = (0.0 - outputLayerNeuron[fileArray][outputNeuron]) * sigmoidFunctionDerivative(outputLayerNeuron[fileArray][outputNeuron]);
                        totalError += (0.0 - outputLayerNeuron[fileArray][outputNeuron]) * (0.0 - outputLayerNeuron[fileArray][outputNeuron]);
                    } 
                    // Calculate delta and totalError if output neuron IS the target
                    else {
                        outputLayerDelta[fileArray][outputNeuron] = (1.0 - outputLayerNeuron[fileArray][outputNeuron]) * sigmoidFunctionDerivative(outputLayerNeuron[fileArray][outputNeuron]);
                        totalError += (1.0 - outputLayerNeuron[fileArray][outputNeuron]) * (1.0 - outputLayerNeuron[fileArray][outputNeuron]);
                    }
                }
                /* Backpropagation from hidden layer */
                for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayerNeuron[fileArray].length; hiddenNeuron++) {
                	// Zero the values from the previous iteration
                    hiddenLayerDelta[fileArray][hiddenNeuron] = 0.0;

                    // Add to the delta for each output neuron
                    for(int outputNeuron = 0; outputNeuron < outputLayerNeuron[fileArray].length; outputNeuron++) {
                        hiddenLayerDelta[fileArray][outputNeuron] += outputLayerDelta[fileArray][outputNeuron] * inputLayerToHiddenLayerWeights[hiddenNeuron][outputNeuron] ;
                    }

                    // Use sigmoid derivative for later weight adjustments
                    hiddenLayerDelta[fileArray][hiddenNeuron] *= sigmoidFunctionDerivative(hiddenLayerNeuron[fileArray][hiddenNeuron]);
                }

                // Weights modification
                temporaryHiddenWeights = Arrays.copyOf(inputLayerToHiddenLayerWeights, inputLayerToHiddenLayerWeights.length);
            	temporaryOutputWeights = Arrays.copyOf(hiddenLayerToOutputLayerWeights, hiddenLayerToOutputLayerWeights.length);

                /* Input to hidden weights */
                for(int inputNeuron = 0; inputNeuron < input[fileArray].length - 1; inputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayerNeuron[fileArray].length; hiddenNeuron++) {
                        inputLayerToHiddenLayerWeights[inputNeuron][hiddenNeuron] +=
                        					(momentum * (inputLayerToHiddenLayerWeights[inputNeuron][hiddenNeuron]
                        					- previousHiddenWeights[inputNeuron][hiddenNeuron]))
                        					+ (learningRate * hiddenLayerDelta[fileArray][hiddenNeuron] * input[fileArray][inputNeuron]);
                    }
                }

                /* Hidden to output weights */
                for(int outputNeuron = 0; outputNeuron < outputLayerNeuron[fileArray].length; outputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayerNeuron[fileArray].length; hiddenNeuron++) {
                    	hiddenLayerToOutputLayerWeights[hiddenNeuron][outputNeuron] +=
                                			(momentum * (hiddenLayerToOutputLayerWeights[hiddenNeuron][outputNeuron]
                                			- previousOutputWeights[hiddenNeuron][outputNeuron]))
                                			+ (learningRate * outputLayerDelta[fileArray][outputNeuron] * hiddenLayerNeuron[fileArray][hiddenNeuron]);
                    }
                }
                
                // Save modified weights as previous for each loop
                previousHiddenWeights = Arrays.copyOf(temporaryHiddenWeights, temporaryHiddenWeights.length);
            	previousOutputWeights = Arrays.copyOf(temporaryOutputWeights, temporaryOutputWeights.length);
                
                // Get total mean squared totalError for epoch
                meanSquaredError += totalError / (TOTAL_OUTPUT_LAYER_NEURONS + 1);
                
                // Reset totalError for next loop
                totalError = 0.0;
            }
            
            // Print the process
            System.out.println("Epoch: " + epochCounter + "error = " + meanSquaredError);
            
            // Check for the epoch count
            if(epochCounter == maxEpochs)
           		break;
            
            // Add epochs
            epochCounter++;
            
            // Save weights into a file after each epoch
            writeToFile("newWeightsHidden.txt", previousHiddenWeights);
            writeToFile("newWeightsHidden.txt", previousOutputWeights);
        }
    } 
    
    /* Method that rounds double values to two decimal point.
     * Requires 1 parameter:
     * 1st parameter is the value of non-decimal number
     */
	public static double roundToTwoDecimals(double value) {
		DecimalFormat df = new DecimalFormat("#.##");
		value = Double.valueOf(df.format(value));
		
		return value;
	}
    
    //  Method getAccuracy calculates the prediction accuracy for chosen 2D input array

    public static void getAccuracy(double[][] data) {
    	double maxValue = 0;
        int correctAnswerPosition = 0;
        double correctPredictionCounter = 0;
        double accuracy = 0;
        
        // Loop through the output layer
        for(int outputNeuron = 0; outputNeuron < outputLayerNeuron.length; outputNeuron++){
            // Reset values for each loop
        	correctAnswerPosition = 0;
            maxValue = 0;
            // Loop through the output elements and check for the maxValue
            for(int layerElement = 0; layerElement < outputLayerNeuron[outputNeuron].length; layerElement++){
            	// Check for the maxValue and save its position
                if(outputLayerNeuron[outputNeuron][layerElement] > maxValue){
                    maxValue = outputLayerNeuron[outputNeuron][layerElement];
                    correctAnswerPosition = layerElement;
                }
            }
            // Count correct predictions
            if(correctAnswerPosition == data[outputNeuron][64])
            	correctPredictionCounter++;
        }
        // Count the accuracy
        accuracy = (correctPredictionCounter / data.length) * 100;
        
        // Print the results
        System.out.println("Total of examples in dataset: " + data.length);
        System.out.println("Nr of correct predictions: " + (int) correctPredictionCounter);
        System.out.println("Accuracy percentage: " + roundToTwoDecimals(accuracy) + "%");   
    }
    
    /* Void type method to print out the results */
    public static void printResults() {
    	System.out.println("First dataset file : \n");
    	calculateDotProducts(firstFileDataArray);
		getAccuracy(firstFileDataArray);
		
		System.out.println("______________________________________\n");
		System.out.println("Second dataset file data:\n");
		calculateDotProducts(secondFileDataArray);
		getAccuracy(secondFileDataArray);

    }
    
	/* Void type method that  starts the whole program. */
	public static void initialise(double[][] input, double learningRate, double leastMeanSquaredError, double momentum, int maxEpochs) {
		storeData();
		
		/* Uncomment this for training process */
		train(input, learningRate, leastMeanSquaredError, momentum, maxEpochs);
		getAccuracy(input);
		
		
		/* Comment this when training */
//		printResults();
	}
	
	public static void main(String[] args) {
		/* parameters are commented above train method */
		initialise(firstFileDataArray, 0.04, 0.01, 0.8, 20);
	}
}
