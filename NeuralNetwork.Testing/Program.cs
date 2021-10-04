using System;
using NeuralNetwork.Core;
using NeuralNetwork.Core.Activations;
using Accord.Math;
using NeuralNetwork.Core.Loss;
using Accord.Statistics;
using NeuralNetwork.Core.ActivationLoss;
using NeuralNetwork.Core.Optimisers;
using System.Linq;
using NeuralNetwork.Core.Optimizers;
using NeuralNetwork.Core.Layers;

namespace NeuralNetwork.Testing
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Algorithm1();
        }

        static void Algorithm1()
        {
            int batchSize = 1000;

            // Generate data
            var (X, y) = SpiralDataset.GenerateSpiralData(batchSize, 3);

            // Init neural network
            var dense1 = new LayerDense(X.Columns(), 16, weightsL2: 5e-4, biasesL2: 5e-4);
            var dropout1 = new LayerDropout(0.2);
            var activation1 = new ActivationReLU();

            var dense2 = new LayerDense(16, 3);
            var lossActivation = new ActivationSoftmaxLossCategoricalCrossentropy();

            //var optimizer = new OptimizerSGD(learningRate: 1, decay: 1e-3, momentum: 0.5);
            //var optimizer = new OptimizerAdaGrad(decay: 1e-4);
            //var optimizer = new OptimizerRMSProp(learningRate: 0.02, decay: 1e-5, rho: 0.999);
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-5);

            // Start Training
            for (int epoch = 0; epoch < 10001; epoch++)
            {
                // Forward pass
                dense1.Forward(X);
                activation1.Forward(dense1.Output);
                dropout1.Forward(activation1.Output);
                dense2.Forward(dropout1.Output);
                double dataLoss = lossActivation.Forward(dense2.Output, y);

                // Calculate regularization penalty
                double regularizationLoss =
                    Loss.RegularizationLoss(dense1) +
                    Loss.RegularizationLoss(dense2);

                // Calculate overall loss
                double loss = dataLoss + regularizationLoss;

                // Backward pass
                lossActivation.Backward(lossActivation.Output, y);
                dense2.Backward(lossActivation.DInputs);
                dropout1.Backward(dense2.DInputs);
                activation1.Backward(dropout1.DInputs);
                dense1.Backward(activation1.DInputs);

                // Update params
                optimizer.PreUpdateParams();
                optimizer.UpdateParams(dense1);
                optimizer.UpdateParams(dense2);
                optimizer.PostUpdateParams();

                // Calculate accuracy
                double accuracy = Loss.Accuracy(lossActivation.Output, y);


                if (epoch % 100 == 0)
                {
                    //Console.Clear();
                    Console.WriteLine($"Epoch: { epoch }");
                    Console.WriteLine($"Data Loss: { dataLoss }");
                    Console.WriteLine($"Reg Loss: { regularizationLoss }");
                    Console.WriteLine($"Loss: { loss }");
                    Console.WriteLine($"Accuracy: { accuracy }");
                    Console.WriteLine($"LR: {optimizer.CurrentLearningRate}");
                    Console.WriteLine();
                    //Console.ReadKey();
                }
            }

            // Validation
            {
                var (XTest, yTest) = SpiralDataset.GenerateSpiralData(100, 3);

                dense1.Forward(XTest);
                activation1.Forward(dense1.Output);
                dense2.Forward(activation1.Output);
                double testLoss = lossActivation.Forward(dense2.Output, yTest);
                double testAccuracy = Loss.Accuracy(lossActivation.Output, yTest);

                Console.WriteLine($"Validation");
                Console.WriteLine($"Loss: { testLoss }");
                Console.WriteLine($"Accuracy: { testAccuracy }");
                Console.WriteLine();
            }
        }
    }
}
