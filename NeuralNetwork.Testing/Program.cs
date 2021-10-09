using System;
using NeuralNetwork.Core;
using NeuralNetwork.Core.Activations;
using Accord.Math;
using NeuralNetwork.Core.Losses;
using Accord.Statistics;
using NeuralNetwork.Core.ActivationLoss;
using NeuralNetwork.Core.Optimizers;
using System.Linq;
using NeuralNetwork.Core.Layers;
using NeuralNetwork.Core.Text;
using NeuralNetwork.Core.Models;
using NeuralNetwork.Core.Accuracies;

namespace NeuralNetwork.Testing
{
    internal class Program
    {
        static void Main(string[] args)
        {
            SpiralModelAlgorithm();
            //Algorithm1();
            //TrainWords();
        }

        static void SpiralModelAlgorithm()
        {
            // Generate data
            var (X, y) = SpiralDataset.GenerateSpiralData(1000, 3);
            var validationData = SpiralDataset.GenerateSpiralData(10, 3);

            // Create model
            var loss = new LossCategoricalCrossentropy();
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-5);
            var accuracy = new AccuracyClassification();
            Model model = new Model(loss, optimizer, accuracy);

            model.Layers.Add(new LayerDense(X.Columns(), 16, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            model.Layers.Add(new LayerDense(16, 3));
            model.Layers.Add(new ActivationSoftmax());

            // Prepare model
            model.Prepare();

            // Train model
            model.Train(X, y, epochs: 1001, logFreq: 100, validationData: validationData);
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

            lossActivation.Loss.TrainableLayers = new[] { dense1, dense2 };

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
                var (dataLoss, regLoss) = lossActivation.Forward(dense2.Output, y);
                double loss = dataLoss + regLoss;

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
                    Console.WriteLine($"Reg Loss: { regLoss }");
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
                var (data_loss, reg_loss) = lossActivation.Forward(dense2.Output, yTest);
                double testLoss = data_loss + reg_loss;
                double testAccuracy = Loss.Accuracy(lossActivation.Output, yTest);

                Console.WriteLine($"Validation");
                Console.WriteLine($"Loss: { testLoss }");
                Console.WriteLine($"Accuracy: { testAccuracy }");
                Console.WriteLine();
            }
        }

        static void TrainWords()
        {
            string cyberbullyingDatasetPath = "D:/Datasets/cyberbullying_parsed_dataset.csv";
            string gloveWordVectorsPath = "D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt";

            // 5,0000ms
            Console.WriteLine("Preparing text reader...");
            var textReader = new TextReaderWordVector(gloveWordVectorsPath);

            // 10,000ms
            Console.WriteLine("Preparing dataset...");
            var dataset = new CyberBullyingDataset(cyberbullyingDatasetPath);
            var (X, y) = dataset.PrepareDataset(textReader, 25);

            // Must find a way to pad vectors before converting to matrix
            Console.WriteLine("Starting algorithm...");

            Console.WriteLine();
        }
    }
}
