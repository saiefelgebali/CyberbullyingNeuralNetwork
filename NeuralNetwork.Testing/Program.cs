﻿using System;
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
            string dataset = "D:/Datasets/cyberbullying/cyberbullying_parsed_dataset.csv";
            string wordvec = "D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt";
            CyberbullyingAlgorithm(dataset, wordvec);

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
            var loss = new LossCategoricalCrossentropy();
            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-5);
            var accuracy = new AccuracyClassification();
            var model = new Model(loss, optimizer, accuracy);

            // Layer 1
            model.Layers.Add(new LayerDense(X.Columns(), 128, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(128, 4));
            model.Layers.Add(new ActivationSoftmax());

            // Prepare model
            model.Prepare();

            // Train model
            model.Train((X, y), (XVal, yVal), epochs: 10, batchSize: 1, logFreq: 0);
        }
    }
}
