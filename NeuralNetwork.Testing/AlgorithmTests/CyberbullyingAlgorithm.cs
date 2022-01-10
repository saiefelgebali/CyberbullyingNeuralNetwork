using Accord.Math;
using NeuralNetwork.Core.MLP.Layers;
using NeuralNetwork.Core.MLP.Accuracies;
using NeuralNetwork.Core.Optimizers;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.MLP.Activations;
using NeuralNetwork.Core.MLP.Losses;
using NeuralNetwork.Core.Text;
using System;

namespace NeuralNetwork.Testing.AlgorithmTests
{
    public class CyberbullyingAlgorithm
    {
        private static readonly int InputLength = 25;

        public static Model CreateCyberbullyingModel(Model model)
        {
            // Layer 1
            model.Layers.Add(new LayerDense(InputLength, 256, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer 2
            model.Layers.Add(new LayerDense(256, 128));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(128, 1));
            model.Layers.Add(new ActivationSigmoid());

            // Prepare model
            model.Prepare();

            return model;
        }

        public static void TestCyberbullyingModel(TextReaderWordVector textReader, string @modelPath)
        {
            // Setup model
            var model = new Model();
            model.Set(loss: new LossCategoricalCrossentropy(), accuracy: new AccuracyClassification());
            model = CreateCyberbullyingModel(model);

            // Use saved params
            model.SetParametersFromFile(modelPath);

            // User input
            var input = "";
            while (input != "QUIT")
            {
                // Gather input
                Console.Write("Enter text (QUIT to stop): ");
                input = Console.ReadLine();

                // Forward pass in model
                var x = textReader.GetWordVectors(input);
                var X = TextReaderWordVector.AverageWordVectors(x);

                if (X.Length == 0)
                {
                    Console.WriteLine("Error: Could not read sentence\n");
                    continue;
                }

                // Evaluate result
                var result = model.Evaluate(X);

                // Output binary result
                if (result.Length == 0) continue;

                var percentCyberbullying = result[0] * 100;
                var percentNeutral = (1 - result[0]) * 100;

                Console.WriteLine($"{percentNeutral}% Neutral");
                Console.WriteLine($"{percentCyberbullying}% Cyberbullying\n");
            }
        }

        public static void TrainCyberbullyingModel(TextReaderWordVector textReader, string @savePath)
        {
            // Get data
            string datasetPath = "D:/Datasets/cyberbullying/cyberbullying_binary_dataset.csv";

            // Prepare dataset
            var dataset = CyberBullyingDataset.PrepareCyberbullyingDataset(datasetPath, textReader);
            var ((X, y), (XVal, yVal)) = dataset;

            // Create model
            var loss = new LossBinaryCrossentropy();
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 1e-5);
            var accuracy = new AccuracyClassification();
            var model = new Model(loss, optimizer, accuracy);
            model = CreateCyberbullyingModel(model);

            // Prepare model
            model.Prepare();

            // Train model
            model.Train((X, y), (XVal, yVal), batchSize: 64, epochs: 200, logFreq: 1000);

            // Save model
            model.SaveParameters(@savePath);
        }
    }
}
