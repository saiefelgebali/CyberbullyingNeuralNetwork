using NeuralNetwork.Core.Text;
using System;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.MLP.Accuracies;
using NeuralNetwork.Core.MLP.Losses;
using NeuralNetwork.Core.MLP.Layers;
using NeuralNetwork.Core.MLP.Activations;
using NeuralNetwork.Core.Optimizers;
using Accord.Math;

namespace NeuralNetwork.Testing.AlgorithmTests
{
    public class SentimentsAlgorithm
    {
        private static Model SetupModel(Model model)
        {
            // Layer 1
            model.Layers.Add(new LayerDense(25, 540, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new ActivationReLU());

            // Layer 2
            model.Layers.Add(new LayerDense(540, 540, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new ActivationReLU());

            // Layer 3
            model.Layers.Add(new LayerDense(540, 540, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(540, 3));
            model.Layers.Add(new ActivationSoftmax());

            model.Prepare();

            return model;
        }

        public static void TestSentimentsModel(TextReaderWordVector textReader, string @modelPath)
        {
            // Setup model
            var model = new Model();
            model.Set(loss: new LossCategoricalCrossentropy(), accuracy: new AccuracyClassification());
            model = SetupModel(model);

            // Prepare model
            model.Prepare();

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

                var classification = result.ArgMax();
                var percentage = result[classification] * 100;

                Console.WriteLine($"class: {classification} at {percentage}%");
                Console.WriteLine($"{result[0] * 100}% Neutral");
                Console.WriteLine($"{result[1] * 100}% Positive");
                Console.WriteLine($"{result[2] * 100}% Negative");
                Console.WriteLine();
            }
        }
        public static Model TrainSentimentsModel(TextReaderWordVector textReader, string @savePath)
        {
            // Specify dataset path
            string datasetPath = "D:/Datasets/twitter_sentiments/twitter_sentiments.csv";

            // Prepare dataset
            var dataset = SentimentsDataset.PrepareSentimentsDataset(datasetPath, textReader);
            var ((X, y), (XVal, yVal)) = dataset;

            // Create model
            var loss = new LossCategoricalCrossentropy();
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-7);
            var accuracy = new AccuracyClassification();
            var model = new Model(loss, optimizer, accuracy);
            model = SetupModel(model);

            // Prepare model
            model.Prepare();

            // Train model
            model.Train((X, y), (XVal, yVal), batchSize: 128, epochs: 5, logFreq: 0);

            // Save model
            model.SaveParameters(@savePath);

            return model;
        }
    }
}
