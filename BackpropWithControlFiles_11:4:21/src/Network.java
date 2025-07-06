import java.util.Scanner;
import java.io.File;
import java.io.PrintWriter;

/** 
 * An A-B-C network that has two connected layers and multiple output activations.
 * It attempts to converge on boolean operator datasets using backpropagation,
 * and it uses a control file to set the network configuration and load in
 * any predetermined weights. 
 * 
 * Author: Amrita Pasupathy
 * Date Created: 11/4/21
*/
public class Network
{
   int numInputActivations;
   int numHiddenActivations; 
   int numOutputActivations;

   double[] A, H, F, T;

   final int NUM_CASES = 4;
   final int NUM_CONNECTIVITY_LAYERS = 2;
   final int THOUSAND = 1000;
   double upperBoundRand, lowerBoundRand;
   boolean train, randomizeWeights;
   double[] currErrors, sumCaseErrors;
   double lambda, maxError, minSumError;
   int iterations, maxIterations;
   long startTime, stopTime;
   File trainingFile, testingFile;

   double[][] inputs, trueOutputs, calcOutputs; 
   double[][][] W;

   double[] omega, psi;
   double[] thetaJ;
   double thetaI, bigOmega;
   double[][][] partialDerivatives, deltaW;


   /**
    * Constructor for the network
    */
   public Network()
   {
      trainingFile = null;
      testingFile = null;

      try
      {
         Scanner input = new Scanner(new File("/Users/23amritap/Documents/vsCode/NN_2021/"
                                                + "BackpropWithControlFiles_11:4:21/src/controlFile.txt"));

            String currLine = input.nextLine();
            train = Boolean.parseBoolean(currLine.substring(0,currLine.indexOf(" ")));

            currLine = input.nextLine();
            randomizeWeights = Boolean.parseBoolean(currLine.substring(0,currLine.indexOf(" ")));

            input.nextLine();

            currLine = input.nextLine();
            numInputActivations = Integer.parseInt(currLine.substring(0,currLine.indexOf(" ")));
   
            currLine = input.nextLine();
            numHiddenActivations = Integer.parseInt(currLine.substring(0,currLine.indexOf(" ")));
   
            currLine = input.nextLine();
            numOutputActivations = Integer.parseInt(currLine.substring(0,currLine.indexOf(" ")));

            inputs = new double[NUM_CASES][2];
            trueOutputs = new double[numOutputActivations][NUM_CASES];

            input.nextLine();

            currLine = input.nextLine();
            upperBoundRand = Double.parseDouble(currLine.substring(0,currLine.indexOf(" ")));
   
            currLine = input.nextLine();
            lowerBoundRand = Double.parseDouble(currLine.substring(0,currLine.indexOf(" ")));
   
            currLine = input.nextLine();
            lambda = Double.parseDouble(currLine.substring(0,currLine.indexOf(" ")));
   
            input.nextLine();
   
            currLine = input.nextLine();
            maxIterations = Integer.parseInt(currLine.substring(0,currLine.indexOf(" ")));
   
            currLine = input.nextLine();
            minSumError = Double.parseDouble(currLine.substring(0,currLine.indexOf(" ")));
   
            for (int t = 0; t < NUM_CASES; t++)
            {
               inputs[t][0] = input.nextDouble();
               inputs[t][1] = input.nextDouble();
               for (int outputNodes = 0; outputNodes < numOutputActivations; outputNodes++)
               {
                  trueOutputs[outputNodes][t] = input.nextDouble();
               }
            }

            input.nextLine();
            input.nextLine();
            input.nextLine();
            trainingFile = new File("/Users/23amritap/Documents/vsCode/NN_2021/BackpropWithControlFiles_11:4:21/src/"
                                       + input.nextLine());

            input.nextLine();
            input.nextLine();
            testingFile = new File("/Users/23amritap/Documents/vsCode/NN_2021/BackpropWithControlFiles_11:4:21/src/"
                                       + input.nextLine());

            input.close();
      } // try

      catch (Exception ex)
      {
         ex.printStackTrace();
      }

      A = new double[numInputActivations];            // array of input activations
      H = new double[numHiddenActivations];           // array of hidden layer activations
      F = new double[numOutputActivations];           // array of output activations
      T = new double[numOutputActivations];           // array of true output values for the current test case

      calcOutputs = new double[numOutputActivations][NUM_CASES];
      sumCaseErrors = new double[NUM_CASES];
      currErrors = new double[numOutputActivations];

      omega = new double[numOutputActivations];
      psi = new double[numOutputActivations];

      thetaJ = new double[numHiddenActivations];

      W = new double[NUM_CONNECTIVITY_LAYERS][][];
      W[0] = new double[numInputActivations][numHiddenActivations];
      W[1] = new double[numHiddenActivations][numOutputActivations];

   } // public Network()


   /**
    * Checks whether the network is set to train or run 
    *
    * @return   true if the network is set to train; otherwise,
    *           false
    */
   public boolean willTrain()
   {
      return train;
   }

   /**
    * Checks whether the network is set to with random weights or predetermined ones
    *
    * @return   true if the network is set to train with random weights; otherwise,
    *           false
    */
   public boolean willRandomizeWeights()
   {
      return randomizeWeights;
   }

   /**
    * Starts timing how long it takes to train the network
    */
   public void startTrainingTimer()
   {
      startTime = System.currentTimeMillis();
   }

   /**
    * Stops timing how long it takes to train the network
    *
    * @return   how long it takes to train the network in milliseconds
    */
   public long stopTrainingTimer()
   {
      stopTime = System.currentTimeMillis();
      return (stopTime - startTime);
   }


   /**
    * Prints a formatted truth table for the provided case
    * 
    * @param testCase       the case whose information is being displayed
    */
   public void printTruthTable(int testCase)
   {
      System.out.print("input 1: " + inputs[testCase][0] + 
         "\tinput 2: " + inputs[testCase][1] + "\ttrue outputs: {");

      for (int i = 0; i < numOutputActivations; i++)
      {
         System.out.print(trueOutputs[i][testCase]);
         if (i != numOutputActivations - 1)  // prints commas after all numbers except the last in the line
         {
            System.out.print(", ");
         }
      }

      System.out.print("}\trun outputs: {");

      for (int i = 0; i < numOutputActivations; i++)
      {
         System.out.print(calcOutputs[i][testCase]);
         if (i != numOutputActivations - 1)  // prints commas after all numbers except the last in the line
         {
            System.out.print(", ");
         }
      }
      System.out.println("}");
   } // public void printTruthTable(int testCase)


   /**
    * Runs the network without training using the predetermined test weights
    */
   public void runNetwork()
   {
      for (int t = 0; t < NUM_CASES; t++)
      {
         A = inputs[t];
         implementNetwork(A);

         for (int setNum = 0; setNum < numOutputActivations; setNum++)
         {
            calcOutputs[setNum][t] = F[setNum];
         }
         
         printTruthTable(t);
      } // for (int t = 0; t < NUM_CASES; t++)
       System.out.println();
   } // public void runNetwork()


   /**
    * Trains the network starting with randomized weights until the largest calculated error 
    * is lower than the threshold error or the maximum number of iterations is reached
    */
   public void trainNetwork()
   {
      startTrainingTimer();

      while (iterations == 0 || (maxError > minSumError && iterations < maxIterations))
      {
         maxError = 0.0;
         for (int t = 0; t < NUM_CASES; t++)
         {
            sumCaseErrors[t] = 0.0;
            A = inputs[t];
            for (int setNum = 0; setNum < numOutputActivations; setNum++)
            {
               T[setNum] = trueOutputs[setNum][t];
            }

            currErrors = implementNetwork(A);
            minimizeError();
            for (int setNum = 0; setNum < numOutputActivations; setNum++)
            {
               calcOutputs[setNum][t] = F[setNum];
               sumCaseErrors[t] += (currErrors[setNum] * currErrors[setNum]);
            }
            
            sumCaseErrors[t] *= 0.5;
         } // for (int t = 0; t < NUM_CASES; t++)
         maxError = calculateMaxError();

         if (iterations % THOUSAND == 1)      // updates the training weights file every thousand iterations
         {
            updateTrainingWeightsFile();
         }
         iterations++;
      } // while (iterations == 0 || (maxError > minSumError && iterations < maxIterations))

      System.out.println("\nTraining Time Taken: " + stopTrainingTimer() + " milliseconds");
      printTrainingInfo();
      updateTrainingWeightsFile();
      for (int i = 0; i < NUM_CASES; i++)
      {
         printTruthTable(i);
      }
      System.out.println();
   } // public void trainNetwork()

   /**
    * Gets the largest calculated case error
    * 
    * @return the maximum error value from all of the cases
    */
   public double calculateMaxError()
   {
      double currMax = 0.0;
      for (int caseNum = 0; caseNum < NUM_CASES; caseNum++)
      {
         currMax = Math.max(currMax, sumCaseErrors[caseNum]);
      }
      return currMax;
   }


   /**
    * Prints information about the network after it finishes training
    */
   public void printTrainingInfo()
   {
      if (iterations == maxIterations)
      {
         System.out.println("Stopping Reason: max iterations reached");
      }

      if (maxError <= minSumError)
      {
         System.out.println("Stopping Reason: maximum calculated error is below minimum threshold");
      }

      System.out.println();
      System.out.println("Network Configuration:");
      System.out.println("\tInput Layer Activations: " + numInputActivations);
      System.out.println("\t1st Hidden Layer Activations: " + numHiddenActivations);
      System.out.println("\tOutput Layer Activations: " + numOutputActivations);
      System.out.println();

      System.out.println("Maximum Iterations: " + maxIterations + "\tIterations Taken: " + iterations);
      System.out.println("Maximum Calculated Error: " + maxError + "\tMinimum Error Threshold: " + minSumError);
      System.out.println("Lambda: " + lambda);
      System.out.println("Random Weight Range: " + lowerBoundRand + " to " + upperBoundRand);
      System.out.println();
   } // public void printTrainingInfo()


   /**
    * Calculates a single randomized weight using the upper and lower bounds
    *
    * @return   a random value inside the range
    */
   public double randomWeight()
   {
      return Math.random() * (upperBoundRand - lowerBoundRand) + lowerBoundRand;
   }


   /**
    * Loads the weights array with random values
    */
   public void loadRandomWeights()
   {
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            W[0][k][j] = randomWeight();
         }
      }

      for (int j = 0; j < numHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            W[1][j][i]  = randomWeight();
         }
      }
   } // public void loadRandomWeights()


   /**
    * Loads the weights array with the stored values from the training file
    */
    public void loadStoredWeights()
    {
      try
      {
         Scanner input = new Scanner(trainingFile);

         for (int k = 0; k < numInputActivations; k++)
         {
            for (int j = 0; j < numHiddenActivations; j++)
            {
               W[0][k][j] = input.nextDouble();
            }
         }

         for (int j = 0; j < numHiddenActivations; j++)
         {
            for (int i = 0; i < numOutputActivations; i++)
            {
               W[1][j][i] = input.nextDouble();
            }
         }
         
         input.close();
      } // try

      catch (Exception ex)
      {
         ex.printStackTrace();
      }
    } // public void loadRandomWeights()


   /**
    * Loads the weights array with the predetermined test weights from the testing file
    */
   public void loadTestWeights()
   {
      try
      {
         Scanner input = new Scanner(testingFile);
         
         for (int k = 0; k < numInputActivations; k++)
         {
            for (int j = 0; j < numHiddenActivations; j++)
            {
               W[0][k][j] = input.nextDouble();
            }
         }
   
         for (int j = 0; j < numHiddenActivations; j++)
         {
            for (int i = 0; i < numOutputActivations; i++)
            {
               W[1][j][i] = input.nextDouble();
            }
         }

         input.close();
      } // try

      catch (Exception ex)
      {
         ex.printStackTrace();
      }
   } // public void loadTestWeights()


   /**
    * Prints the training weights back into the training file
    */
    public void updateTrainingWeightsFile()
    {
      try
      {
         PrintWriter printWriter = new PrintWriter(trainingFile);

         for (int k = 0; k < numInputActivations; k++)
         {
            for (int j = 0; j < numHiddenActivations; j++)
            {
               printWriter.print(W[0][k][j] + " ");
            }
            printWriter.println();
         }
         printWriter.println();         // empty line for formatting reasons
   
         for (int j = 0; j < numHiddenActivations; j++)
         {
            for (int i = 0; i < numOutputActivations; i++)
            {
               printWriter.print(W[1][j][i] + " ");
            }
            printWriter.println();
         }
         printWriter.close (); 
      } // try

      catch (Exception ex)
      {
         ex.printStackTrace();
      }
    } // public void loadTestWeights()


   /**
    * Evaluates the network to calculate the output values for each dataset
    *
    * @param A   an array of activation inputs
    * @return   an array of the calculated outputs
    */
   public double[] implementNetwork(double[] A)
   {
      for (int i = 0; i < numOutputActivations; i++)
      {
         thetaI = 0.0;
         for (int j = 0; j < numHiddenActivations; j++)
         {
            thetaJ[j] = 0.0;
            for (int k = 0; k < numInputActivations; k++)
            {
               thetaJ[j] += W[0][k][j] * A[k];
            }

            H[j] = threshold(thetaJ[j]);
            thetaI += W[1][j][i] * H[j];
         } // for (int j = 0; j < numHiddenActivations; j++)

         F[i] = threshold(thetaI);
         omega[i] = T[i] - F[i];
         psi[i] = omega[i] * thresholdDerivative(thetaI);
      } // for (int i = 0; i < numOutputActivations; i++)
      return omega;

   } // public double[] implementNetwork(double[] A)


   /**
    * Applies the threshold function (currently sigmoid) to a value
    *
    * @param theta   the inputted value on which to apply the function
    * @return  the output of the threshold function
    */
   public double threshold(double theta)
   {
      return 1.0/(1.0 + Math.exp(-theta));
   }


   /**
    * Applies the derivative of the threshold function (currently sigmoid) to a value
    *
    * @param theta   the inputted value on which to apply the function
    * @return  the output of the derivative of the threshold function
    */
   public double thresholdDerivative(double theta)
   {
      double sigmoidTheta = threshold(theta);
      return sigmoidTheta * (1.0 - sigmoidTheta);
   }


   /**
    * Uses the network's output values to calculate the change in weights using gradient descent
    *
    */
   public void minimizeError()
   {
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            bigOmega = 0.0;
            for (int i = 0; i < numOutputActivations; i++)
            {
               bigOmega += psi[i] * W[1][j][i];
               W[1][j][i] += lambda * H[j] * psi[i];
            }
            W[0][k][j] += lambda * A[k] * bigOmega * thresholdDerivative(thetaJ[j]);

         } // for (int j = 0; j < numHiddenActivations; j++)
      } // for (int k = 0; k < numInputActivations; k++)
   } // public void minimizeError()


   /**
    * Creates a network and either runs or trains it
    * 
    * @param args   the input array of information from the command line
    */
   public static void main(String[] args)
   {
      Network network = new Network();
       
      if (network.willTrain())
      {
         if (network.willRandomizeWeights())
         {
            network.loadRandomWeights();
         }
         else
         {
            network.loadStoredWeights();
         }
         network.trainNetwork();

      } // if (network.willTrain())
      else
      {
         network.loadTestWeights();
         network.runNetwork();
      }
      
   } // public static void main(String[] args)
} // public class Network