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
using NeuralNetwork.Core.Accuracies;

namespace NeuralNetwork.Testing
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string dataset = "D:/Datasets/cyberbullying/cyberbullying_binary_dataset.csv";
            string wordvec = "D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt";
            CyberbullyingAlgorithm(dataset, wordvec);
            //SpiralAlgorithm();
            //SpiralDataset.SpiralModelAlgorithm();

        }
        static void CyberbullyingAlgorithm(string datasetPath, string wordvecPath)
        {
            // Prepare dataset
            var textReader = new TextReaderWordVector(wordvecPath);
            var dataset = CyberBullyingDataset.PrepareCyberbullyingDataset(datasetPath, textReader);
            var ((X, y), (XVal, yVal)) = dataset;

            // Normalize data
            double maxValue = X.Abs().Max();
            X = X.Divide(maxValue);
            XVal = XVal.Divide(maxValue);

            // Create model
            var loss = new LossBinaryCrossentropy();
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-5);
            var accuracy = new AccuracyClassification();
            var model = new Model(loss, optimizer, accuracy);

            // Layer 1
            model.Layers.Add(new LayerDense(X.Columns(), 256, weightsL2: 5e-4, biasesL2: 5e-4));
            //model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(256, 1));
            model.Layers.Add(new ActivationSigmoid());

            // Prepare model
            model.Prepare();

            // Train model
            model.Train((X, y), (XVal, yVal), batchSize: 128, epochs: 500, logFreq: 1000);

            // Allow user to test
            while (true)
            {
                // Get input
                Console.Write("Enter text: ");
                string text = Console.ReadLine();

                // Convert to wordvec
                double[][] vector = textReader.GetWordVectors(text);

                // Normalize
                vector = vector.Divide(maxValue);

                if (vector.Length == 0)
                {
                    Console.WriteLine("Could not parse text.");
                    continue;
                }

                // Apply sample to batch
                double[][] XTest = new double[][] { TextReaderWordVector.CombineWordVectors(vector) };

                // Forward pass
                double[][] output = model.Evaluate(XTest);

                Console.WriteLine($"Cyberbullying: {output[0][0]}");
                Console.WriteLine($"Neutral: {1 - output[0][0]}");
            }
        } 
        
        static void SpiralAlgorithm()
        {
            // Prepare dataset
            var (X, y) = SpiralDataset.GenerateSpiralData(1000, 3);
            var (XVal, yVal) = SpiralDataset.GenerateSpiralData(10, 3);

            // Normalize data
            double maxValue = X.Abs().Max();
            X = X.Divide(maxValue);
            XVal = XVal.Divide(maxValue);

            // Create model
            var loss = new LossCategoricalCrossentropy();
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-5);
            var accuracy = new AccuracyClassification();
            var model = new Model(loss, optimizer, accuracy);

            // Layer 1
            model.Layers.Add(new LayerDense(X.Columns(), 32, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(32, 3));
            model.Layers.Add(new ActivationSoftmax());

            // Prepare model
            model.Prepare();

            // Train model
            model.Train((X, y), (XVal, yVal), epochs: 10, batchSize: 500, logFreq: 1000);
        }
    }
}
