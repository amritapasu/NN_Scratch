import java.util.Scanner;
import java.io.File;
import java.io.PrintWriter;

/** 
 * An A-B-C-D network that has three connected layers and multiple output activations.
 * It attempts to converge on boolean operator datasets using backpropagation,
 * and it uses a control file to set the network configuration and load in
 * any predetermined weights. 
 * 
 * Author: Amrita Pasupathy
 * Date Created: 11/11/21
*/
public class Network
{
   int numInputActivations; 
   int numFirstHiddenActivations;
   int numSecondHiddenActivations;
   int numOutputActivations;

   double[][] activations, theta;
   double[] T;

   final int NUM_CASES = 4;
   final int N_LAYERS = 4;
   final int NUM_CONNECTIVITY_LAYERS = N_LAYERS - 1;
   final int THOUSAND = 1000;
   double upperBoundRand, lowerBoundRand;
   boolean train, randomizeWeights;
   double[] sumCaseErrors;
   double lambda, maxError, minSumError, currError;
   int iterations, maxIterations;
   long startTime, stopTime;
   File trainingFile, testingFile;
   String mainControlFileName;

   double[][] inputs, trueOutputs, calcOutputs;
   double[][][] W;

   double[] psi;
   double[] bigOmegaJ, bigOmegaK, bigPsiJ, bigPsiK;
   double thetaI, thetaJ, thetaK;


   /**
    * Constructor for the network
    *
    * @param controlFile       the control file name being passed on the command line
    */
   public Network(String controlFile)
   {
      trainingFile = null;
      testingFile = null;

      if (controlFile.equals(""))
      {
         mainControlFileName = "controlFileDefault.txt";
      }
      else
      {
         mainControlFileName = controlFile;
      }

      try
      {
         Scanner input = new Scanner(new File ("/Users/23amritap/Documents/vsCode/NN_2021/"
         + "A-B-C-D_Backprop_11:11:21/src/" + mainControlFileName));

         String currLine = input.nextLine();
         train = Boolean.parseBoolean(currLine.substring(0,currLine.indexOf(" ")));

         currLine = input.nextLine();
         randomizeWeights = Boolean.parseBoolean(currLine.substring(0,currLine.indexOf(" ")));

         input.nextLine();

         currLine = input.nextLine();
         numInputActivations = Integer.parseInt(currLine.substring(0,currLine.indexOf(" ")));
   
         currLine = input.nextLine();
         numFirstHiddenActivations = Integer.parseInt(currLine.substring(0,currLine.indexOf(" ")));

         currLine = input.nextLine();
         numSecondHiddenActivations = Integer.parseInt(currLine.substring(0,currLine.indexOf(" ")));
   
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
         } // for (int t = 0; t < NUM_CASES; t++)

         input.nextLine();
         input.nextLine();
         input.nextLine();
         trainingFile = new File("/Users/23amritap/Documents/vsCode/NN_2021/A-B-C-D_Backprop_11:11:21/src/"
                                       + input.nextLine());

         input.nextLine();
         input.nextLine();
         testingFile = new File("/Users/23amritap/Documents/vsCode/NN_2021/A-B-C-D_Backprop_11:11:21/src/"
                                       + input.nextLine());

         input.close();
      } // try

      catch (Exception ex)
      {
         ex.printStackTrace();
      }

      activations = new double[N_LAYERS][];
      activations[0] = new double[numInputActivations];           // array of input activations
      activations[1] = new double[numFirstHiddenActivations];     // array of 1st hidden layer activations
      activations[2] = new double[numSecondHiddenActivations];    // array of 2nd hidden layer activations
      activations[3] = new double[numOutputActivations];          // array of output activations

      calcOutputs = new double[numOutputActivations][NUM_CASES];
      T = new double[numOutputActivations];           // array of true output values for the current test case

      W = new double[NUM_CONNECTIVITY_LAYERS][][];
      W[0] = new double[numInputActivations][numFirstHiddenActivations];
      W[1] = new double[numFirstHiddenActivations][numSecondHiddenActivations];
      W[2] = new double[numSecondHiddenActivations][numOutputActivations];

      if (train)
      {
         sumCaseErrors = new double[NUM_CASES];

         bigOmegaJ = new double[numSecondHiddenActivations];
         bigOmegaK = new double[numFirstHiddenActivations];
         bigPsiJ = new double[numSecondHiddenActivations];
         bigPsiK = new double[numFirstHiddenActivations];

         psi = new double[numOutputActivations];

         theta = new double[N_LAYERS - 1][];
         theta[0] = new double[numFirstHiddenActivations];
         theta[1] = new double[numSecondHiddenActivations];
         theta[2] = new double[numOutputActivations];

      } // if (train)
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
         activations[0] = inputs[t];
         implementNetworkRun(activations[0]);

         for (int setNum = 0; setNum < numOutputActivations; setNum++)
         {
            calcOutputs[setNum][t] = activations[3][setNum];
         }
         
         printTruthTable(t);
      } // for (int t = 0; t < NUM_CASES; t++)
       System.out.println();
   } // public void runNetwork()


   /**
    * Trains the network until the largest calculated error is lower than the threshold error
    * or the maximum number of iterations is reached
    */
   public void trainNetwork()
   {
      startTrainingTimer();
      for (int t = 0; t < NUM_CASES; t++)     // evaluates the network for the first time with the original weights
         {
            activations[0] = inputs[t];
            for (int setNum = 0; setNum < numOutputActivations; setNum++)
            {
               T[setNum] = trueOutputs[setNum][t];
            }

            implementNetworkTrain(activations[0]);
            for (int setNum = 0; setNum < numOutputActivations; setNum++)
            {
               currError = T[setNum] - activations[3][setNum];
               calcOutputs[setNum][t] = activations[3][setNum];
               sumCaseErrors[t] += currError * currError;
            }
            sumCaseErrors[t] *= 0.5;
         } // for (int t = 0; t < NUM_CASES; t++)

      while (iterations == 0 || (maxError > minSumError && iterations < maxIterations))
      {
         maxError = 0.0;
         for (int t = 0; t < NUM_CASES; t++)
         {
            minimizeError();
            sumCaseErrors[t] = 0.0;
            activations[0] = inputs[t];

            for (int setNum = 0; setNum < numOutputActivations; setNum++)
            {
               T[setNum] = trueOutputs[setNum][t];
            }

            implementNetworkTrain(activations[0]);
            for (int setNum = 0; setNum < numOutputActivations; setNum++)
            {
               currError = T[setNum] - activations[3][setNum];
               calcOutputs[setNum][t] = activations[3][setNum];
               sumCaseErrors[t] += currError * currError;
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

      for (int t = 0; t < NUM_CASES; t++)     // runs with the final weights
      {
         activations[0] = inputs[t];
         implementNetworkRun(activations[0]);

         for (int setNum = 0; setNum < numOutputActivations; setNum++)
         {
            calcOutputs[setNum][t] = activations[3][setNum];
         }
         
         printTruthTable(t);
      } // for (int t = 0; t < NUM_CASES; t++)

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
      System.out.println("Control File: " + mainControlFileName);
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
      System.out.println("\t1st Hidden Layer Activations: " + numFirstHiddenActivations);
      System.out.println("\t1st Hidden Layer Activations: " + numSecondHiddenActivations);
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
      for (int m = 0; m < numInputActivations; m++)
      {
         for (int k = 0; k < numFirstHiddenActivations; k++)
         {
            W[0][m][k] = randomWeight();
         }
      }
         
      for (int k = 0; k < numFirstHiddenActivations; k++)
      {
         for (int j = 0; j < numSecondHiddenActivations; j++)
         {
            W[1][k][j] = randomWeight();
         }
      }
   
      for (int j = 0; j < numSecondHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            W[2][j][i] = randomWeight();
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

         for (int m = 0; m < numInputActivations; m++)
         {
            for (int k = 0; k < numFirstHiddenActivations; k++)
            {
               W[0][m][k] = input.nextDouble();
            }
         }
         
         for (int k = 0; k < numFirstHiddenActivations; k++)
         {
            for (int j = 0; j < numSecondHiddenActivations; j++)
            {
               W[1][k][j] = input.nextDouble();
            }
         }
   
         for (int j = 0; j < numSecondHiddenActivations; j++)
         {
            for (int i = 0; i < numOutputActivations; i++)
            {
               W[2][j][i] = input.nextDouble();
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

         for (int m = 0; m < numInputActivations; m++)
         {
            for (int k = 0; k < numFirstHiddenActivations; k++)
            {
               W[0][m][k] = input.nextDouble();
            }
         }
         
         for (int k = 0; k < numFirstHiddenActivations; k++)
         {
            for (int j = 0; j < numSecondHiddenActivations; j++)
            {
               W[1][k][j] = input.nextDouble();
            }
         }
   
         for (int j = 0; j < numSecondHiddenActivations; j++)
         {
            for (int i = 0; i < numOutputActivations; i++)
            {
               W[2][j][i] = input.nextDouble();
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

         for (int m = 0; m < numInputActivations; m++)
         {
            for (int k = 0; k < numFirstHiddenActivations; k++)
            {
               printWriter.print(W[0][m][k] + " ");
            }
            printWriter.println();
         }
         printWriter.println();
         
         for (int k = 0; k < numFirstHiddenActivations; k++)
         {
            for (int j = 0; j < numSecondHiddenActivations; j++)
            {
               printWriter.print(W[1][k][j] + " ");
            }
            printWriter.println();
         }
         printWriter.println();
   
         for (int j = 0; j < numSecondHiddenActivations; j++)
         {
            for (int i = 0; i < numOutputActivations; i++)
            {
               printWriter.print(W[2][j][i] + " ");
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
    * Evaluates the network to calculate the output values for each dataset when training
    *
    * @param A   an array of activation inputs
    */
   public void implementNetworkTrain(double[] A)
   {
      for (int i = 0; i < numOutputActivations; i++)
      {
         theta[2][i] = 0.0;
         for (int j = 0; j < numSecondHiddenActivations; j++)
         {
            theta[1][j] = 0.0;
            for (int k = 0; k < numFirstHiddenActivations; k++)
            {
               theta[0][k] = 0.0;
               for (int m = 0; m < numInputActivations; m++)
               {
                  theta[0][k] += W[0][m][k] * activations[0][m];
               } // for (int m = 0; m < numInputActivations; m++)

               activations[1][k] = threshold(theta[0][k]);
               theta[1][j] += W[1][k][j] * activations[1][k];
            } // for (int k = 0; k < numFirstHiddenActivations; k++)

            activations[2][j] = threshold(theta[1][j]);
            theta[2][i] += W[2][j][i] * activations[2][j];
         } // for (int j = 0; j < numSecondHiddenActivations; j++)

         activations[3][i] = threshold(theta[2][i]);
         psi[i] = (T[i] - activations[3][i]) * thresholdDerivative(theta[2][i]);
      } // for (int i = 0; i < numOutputActivations; i++)

   } // public void implementNetworkTrain(double[] A)
   
/**
    * Evaluates the network to calculate the output values for each dataset when running
    *
    * @param A   an array of activation inputs
    */
    public void implementNetworkRun(double[] A)
    {
       for (int i = 0; i < numOutputActivations; i++)
       {
          thetaI = 0.0;
          for (int j = 0; j < numSecondHiddenActivations; j++)
          {
             thetaJ = 0.0;
             for (int k = 0; k < numFirstHiddenActivations; k++)
             {
                thetaK = 0.0;
                for (int m = 0; m < numInputActivations; m++)
                {
                   thetaK += W[0][m][k] * activations[0][m];
                } // for (int m = 0; m < numInputActivations; m++)
 
                activations[1][k] = threshold(thetaK);
                thetaJ += W[1][k][j] * activations[1][k];
             } // for (int k = 0; k < numFirstHiddenActivations; k++)
 
             activations[2][j] = threshold(thetaJ);
             thetaI += W[2][j][i] * activations[2][j];
          } // for (int j = 0; j < numSecondHiddenActivations; j++)
 
          activations[3][i] = threshold(thetaI);
       } // for (int i = 0; i < numOutputActivations; i++)
 
    } // public void implementNetworkRun(double[] A)

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
    * Uses the network's output values to calculate the change in weights using backpropagation
    *
    */
   public void minimizeError()
   {
      for (int j = 0; j < numSecondHiddenActivations; j++)
      {
         bigOmegaJ[j] = 0.0;
         for (int i = 0; i < numOutputActivations; i++)
         {
            bigOmegaJ[j] += psi[i] * W[2][j][i];
            W[2][j][i] += lambda * activations[2][j] * psi[i];
         }
         bigPsiJ[j] = bigOmegaJ[j] * thresholdDerivative(theta[1][j]);
      } // for (int j = 0; j < numSecondHiddenActivations; j++)

      for (int k = 0; k < numFirstHiddenActivations; k++)
      {
         bigOmegaK[k] = 0.0;
         for (int j = 0; j < numSecondHiddenActivations; j++)
         {
            bigOmegaK[k] += bigPsiJ[j] * W[1][k][j];
            W[1][k][j] += lambda * activations[1][k] * bigPsiJ[j];
         }
         bigPsiK[k] = bigOmegaK[k] * thresholdDerivative(theta[0][k]);
      } // for (int k = 0; k < numFirstHiddenActivations; k++)

      for (int m = 0; m < numInputActivations; m++)
      {
         for (int k = 0; k < numFirstHiddenActivations; k++)
         {
            W[0][m][k] += activations[0][m] * bigPsiK[k];
         }
      } // for (int m = 0; m < numInputActivations; m++)
   } // public void minimizeError()


   /**
    * Creates a network based on the information in a control file and either runs or trains it
    * 
    * @param args   the input array of information from the command line
    */
   public static void main(String[] args)
   {
      String controlFileName = "";
      if (args.length != 0)
      {
         controlFileName = args[0];
      }

      Network network = new Network(controlFileName);
       
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