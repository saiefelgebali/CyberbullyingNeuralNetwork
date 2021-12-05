using NeuralNetwork.Core.Text;
using System;
using System.Linq;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.Accuracies;
using NeuralNetwork.Core.Losses;
using NeuralNetwork.Core.Layers;
using NeuralNetwork.Core.Activations;
using NeuralNetwork.Core.Optimizers;
using Accord.Math;
using Microsoft.VisualBasic.FileIO;
using System.Collections.Generic;
using NeuralNetwork.Core;

namespace NeuralNetwork.Testing
{
    internal class TwitterSentiments
    {
        private static Model SetupModel(Model model)
        {
            // Layer 1
            model.Layers.Add(new LayerDense(25, 128, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer 2
            model.Layers.Add(new LayerDense(128, 128, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());
            
            // Layer 3
            model.Layers.Add(new LayerDense(128, 64, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(64, 3));
            model.Layers.Add(new ActivationSoftmax());

            return model;
        }

        public static void TestSentimentsModel(TextReaderWordVector textReader, string @modelPath)
        {
            // Setup model
            var model = new Model();
            model.Set(loss: new LossCategoricalCrossentropy(), accuracy: new AccuracyClassification());
            SetupModel(model);

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
                var X = textReader.GetCombinedWordVectors(input);
                X = textReader.NormalizeWordVectors(X);
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
            string trainPath = "D:/Datasets/twitter_sentiment_analysis/twitter_training.csv";
            string valPath = "D:/Datasets/twitter_sentiment_analysis/twitter_validation.csv";

            // Prepare dataset
            var dataset = PrepareSentimentsDataset(trainPath, valPath, textReader);
            var ((X, y), (XVal, yVal)) = dataset;

            // Normalize data
            X = textReader.NormalizeWordVectors(X);
            XVal = textReader.NormalizeWordVectors(XVal);

            // Create model
            var loss = new LossCategoricalCrossentropy();
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-7);
            var accuracy = new AccuracyClassification();
            var model = new Model(loss, optimizer, accuracy);
            SetupModel(model);

            // Prepare model
            model.Prepare();

            // Train model
            model.Train((X, y), (XVal, yVal), batchSize: 100, epochs: 500, logFreq: 100);

            // Save model
            model.SaveParameters(@savePath);

            return model;
        }

        static ((double[][], int[]), (double[][], int[])) PrepareSentimentsDataset(string trainPath, string valPath, TextReaderWordVector textReader)
        {
            var (XTrain, yTrain) = ParseSentimentsDataset(trainPath, textReader);

            var shuffle = Utility.ShuffleIndices(XTrain.Length);
            XTrain = Utility.ShuffleArray(XTrain, shuffle);
            yTrain = Utility.ShuffleArray(yTrain, shuffle);

            var(XVal, yVal) = ParseSentimentsDataset(valPath, textReader);

            // Return as tuple of tuples
            return ((XTrain, yTrain), (XVal, yVal));
        }

        static (double[][], int[]) ParseSentimentsDataset(string path, TextReaderWordVector textReader)
        {
            // Start reading from dataset csv
            using var csvParser = new TextFieldParser(@path);

            csvParser.CommentTokens = new string[] { "#" };
            csvParser.SetDelimiters(new string[] { "," });
            csvParser.HasFieldsEnclosedInQuotes = true;

            // Skip column names
            csvParser.ReadLine();

            // Samples
            var sampleTextList = new List<double[]>();
            var sampleTargetList = new List<int>();

            while (!csvParser.EndOfData)
            {
                // Read each row's fields
                string[] fields;
                try
                {
                    fields = csvParser.ReadFields();
                }
                catch
                {
                    // Skip unreadable lines
                    continue;
                }

                // Validate field length
                if (fields.Length != 4) continue;

                // Extract sample data
                string text = fields[3];
                string sentiment = fields[2];
                int targetClass;
                switch (sentiment)
                {
                    case "Neutral":
                        targetClass = 0;
                        break;
                    case "Positive":
                        targetClass = 1;
                        break;
                    case "Negative":
                        targetClass = 2;
                        break;
                    default:
                        targetClass = 0;
                        break;
                }

                // Get vector
                double[] textVector = TextReaderWordVector.CombineWordVectors(textReader.GetWordVectors(text));

                // Validate text
                if (textVector.Length == 0)
                {
                    continue;
                }

                // Add sample to list
                sampleTextList.Add(textVector);

                // Add sample target to list
                sampleTargetList.Add(targetClass);
            }

            return (sampleTextList.ToArray(), sampleTargetList.ToArray());
        }
    }
}
