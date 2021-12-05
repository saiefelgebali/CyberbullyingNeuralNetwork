using System;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.Activations;
using Accord.Math;
using NeuralNetwork.Core.Losses;
using NeuralNetwork.Core.Optimizers;
using System.Linq;
using NeuralNetwork.Core.Layers;
using NeuralNetwork.Core.Text;
using NeuralNetwork.Core.Accuracies;
using System.Text.Json;
using System.IO;

namespace NeuralNetwork.Testing
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var wordvecModel = "D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt";
            var textReader = new TextReaderWordVector(wordvecModel);

            // Train a cyberbullying model
            //var savePath = "D:/Projects/ml_models/cyberbullying_model.json";
            //CyberbullyingAlgorithm.TrainCyberbullyingModel(textReader, savePath);

            // Train a sentiment analysis model
            //var savePath = "D:/Projects/ml_models/twitter_sentiments_model_2.json";
            //var sentimentsModel = SentimentsAlgorithm.TrainSentimentsModel(textReader, savePath);
            //SentimentsAlgorithm.TestSentimentsModel(textReader, savePath);

            // Train on a new dataset
            //var savePath = "D:/Projects/ml_models/twitter_sentiments_model_3.json";
            //TwitterSentiments.TrainSentimentsModel(textReader, savePath);
            //TwitterSentiments.TestSentimentsModel(textReader, savePath);
        }

        static void TestCyberbullyingModel()
        {
            string path = "D:/Projects/ml_models/cyberbullying_model.json";

            var model = new Model();

            // Layer 1
            model.Layers.Add(new LayerDense(256, 256));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(256, 1));
            model.Layers.Add(new ActivationSigmoid());

            // Loss
            model.Set(loss: new LossBinaryCrossentropy());

            model.Prepare();

            model.SetParametersFromFile(path);

            return;
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

            // Save model
            model.SaveParameters("D:/Projects/ml_models/spiral_model.json");
        }
    }
}
