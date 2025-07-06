/** 
 * An A-B-C network that has two connected layers and multiple output activations,
 * and it attempts to converge on boolean operator datasets using backpropagation.
 * 
 * Author: Amrita Pasupathy
 * Date Created: 10/25/21
*/
public class Network
{
   int numInputActivations;  // fix return types on implement and minimize (switched), training timer documentation on both
   int numHiddenActivations; // add timer to A-B-C network, check how to do control file for weights in class
   int numOutputActivations; // check just running the network with a new test array, check if can use longs

   double[] A, H, F, T; 

   final int NUM_CASES = 4;
   final int NUM_CONNECTIVITY_LAYERS = 2;
   double upperBoundRand, lowerBoundRand;
   boolean train;
   double[] currErrors, sumCaseErrors;
   double lambda, maxError, minSumError;
   int iterations, maxIterations;
   long startTime, stopTime;

   /*
   * To train or run the network with another set of inputs,
   * change the values in this 2D array
   */
   double[][] inputs = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};

   /*
   * To train or run the network with another set of outputs, 
   * add another set of outputs in the form of a 1D array to this 2D array
   * 
   */
   double[][] trueOutputs= {{0.0, 0.0, 0.0, 1.0}, {0.0, 1.0, 1.0, 1.0}, {0.0, 1.0, 1.0, 0.0}};

   double[][] calcOutputs; 

   double[][][] W;

   /*
   * Although this array is currently initialized to be specific to the the 2-2-1 configuration,
   * it can be manually changed to match different configurations
   */
   double[][][] testWeights = {{{0.1, 0.2}, {0.3, 0.4}}, {{0.5}, {0.6}}};      

   double[] omega, psi;
   double[] thetaJ, thetaI;
   double[] bigOmega, bigPsi;
   double[][][] partialDerivatives, deltaW;


   /**
    * Constructor for the network
    * 
    * @param isTrain                   whether the network will train or run
    * @param hiddenActivations         the number of hidden layer activations
    * @param outputActivations         the number of output layer activations
    * @param upperBound                the upper bound of the random weight range
    * @param lowerBound                the upper bound of the random weight range
    * @param lam                       the learning rate (lambda)
    * @param maxIter                   the maximum number or iterations before the network stops training
    * @param minErr                    the largest error needed before the network stops training
    */
   public Network(boolean isTrain, int hiddenActivations, int outputActivations, 
   double upperBound, double lowerBound, double lam, int maxIter, double minErr)
   {
      numInputActivations = 2;                        // should not be changed at this stage
      numHiddenActivations = hiddenActivations;
      numOutputActivations = outputActivations;

      A = new double[numInputActivations];            // array of input activations
      H = new double[numHiddenActivations];               // array of hidden layer activations
      F = new double[numOutputActivations];           // array of output activations

      upperBoundRand = upperBound;
      lowerBoundRand = lowerBound;

      train = isTrain;
      lambda = lam;
      minSumError = minErr;
      maxIterations = maxIter;

      T = new double[numOutputActivations];           // array of true output values for the current test case

      calcOutputs = new double[numOutputActivations][NUM_CASES];
      sumCaseErrors = new double[NUM_CASES];
      currErrors = new double[numOutputActivations];

      omega = new double[numOutputActivations];
      psi = new double[numOutputActivations];

      bigOmega = new double[numHiddenActivations];
      bigPsi = new double[numHiddenActivations];

      thetaJ = new double[numHiddenActivations];
      thetaI = new double[numOutputActivations];

      W = new double[NUM_CONNECTIVITY_LAYERS][][];
      W[0] = new double[numInputActivations][numHiddenActivations];
      W[1] = new double[numHiddenActivations][numOutputActivations];

      partialDerivatives = new double[NUM_CONNECTIVITY_LAYERS][][];
      partialDerivatives[0] = new double[numInputActivations][numHiddenActivations];
      partialDerivatives[1] = new double[numHiddenActivations][numOutputActivations];

      deltaW = new double[NUM_CONNECTIVITY_LAYERS][][];
      deltaW[0] = new double[numInputActivations][numHiddenActivations];
      deltaW[1] = new double[numHiddenActivations][numOutputActivations];
   } // public Network(boolean isTrain, int hiddenActivations, int outputActivations, 
     //   double upperBound, double lowerBound, double lam, int maxIter, double minErr)


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
    * Se
    */
   public void startTrainingTimer()
   {
      startTime = System.currentTimeMillis();
   }

   /**
    * Checks whether the network is set to train or run 
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
         if (i != numOutputActivations - 1)
         {
            System.out.print(", ");
         }
       }

      System.out.print("}\trun outputs: {");

      for (int i = 0; i < numOutputActivations; i++)
      {
         System.out.print(calcOutputs[i][testCase]);
         if (i != numOutputActivations - 1)
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
      loadTestWeights();

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
      loadRandomWeights();
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
         iterations++;
         maxError = calculateMaxError();
      } // while (iterations == 0 || (maxError > minSumError && iterations < maxIterations))

      System.out.println("\nTraining Time Taken: " + stopTrainingTimer() + " milliseconds");
      printTrainingInfo();
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
    * Loads the weights array with the predetermined test weights
    */
   public void loadTestWeights()
   {
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            W[0][k][j] = testWeights[0][k][j];
         }
      }

      for (int j = 0; j < numHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            W[1][j][i]  = testWeights[1][j][0];
         }
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
         thetaI[i] = 0.0;
         for (int j = 0; j < numHiddenActivations; j++)
         {
            thetaJ[j] = 0.0;
            for (int k = 0; k < numInputActivations; k++)
            {
               thetaJ[j] += W[0][k][j] * A[k];
            }

            H[j] = threshold(thetaJ[j]);
            thetaI[i] += W[1][j][i] * H[j];
         }

         F[i] = threshold(thetaI[i]);
         omega[i] = T[i] - F[i];
         psi[i] = omega[i] * thresholdDerivative(thetaI[i]);
      }
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
    * @return   an array of errors for this case 
    */
   public void minimizeError()
   {
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            bigOmega[j] = 0.0;
            for (int i = 0; i < numOutputActivations; i++)
            {
               bigOmega[j] += psi[i] * W[1][j][i];
               deltaW[1][j][i] = lambda * H[j] * psi[i];
               W[1][j][i] += deltaW[1][j][i];
            }
            bigPsi[j] = bigOmega[j] * thresholdDerivative(thetaJ[j]);

            deltaW[0][k][j] = lambda * A[k] * bigPsi[j];
            W[0][k][j] += deltaW[0][k][j];
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
      Network network = new Network(true, 500, 3, 1.5, 0.1, 0.3, 20000, 0.001);
      /*
       * Constructor for the Network class:
       *    Network.Network(boolean isTrain, int hiddenActivations, int outputActivations, 
       *      double upperBound, double lowerBound, double lam, int maxIter, double minErr)
       * 
       * @param isTrain                   whether the network will train or run
       * @param hiddenActivations         the number of hidden layer activations
       * @param outputActivations         the number of output layer activations
       * @param upperBound                the upper bound of the random weight range
       * @param lowerBound                the upper bound of the random weight range
       * @param lam                       the learning rate (lambda)
       * @param maxIter                   the maximum number or iterations before the network stops training
       * @param minErr                    the largest error needed before the network stops training
       * 
       * to train or run the network with another set of inputs or outputs 
       *    that don't necessarily have to be boolean operators,
       *    make changes to the instance variable declarations at the top of the class
       *    and then change this parameter to match the index of the new set of outputs
       * 
       */

      if (network.willTrain())
      {
         network.trainNetwork();
      }
      else
      {
         network.runNetwork();
      }
   } // public static void main(String[] args)
} // public class Network