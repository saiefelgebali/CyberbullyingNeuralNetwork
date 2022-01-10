using System;
using Accord.Math;
using System.Linq;
using NeuralNetwork.Core.Text;
using System.Text.Json;
using System.IO;
using NeuralNetwork.Testing;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.MLP;
using NeuralNetwork.Core.MLP.Layers;
using NeuralNetwork.Core.MLP.ActivationLoss;
using NeuralNetwork.Core.MLP.Accuracies;
using NeuralNetwork.Core.MLP.Activations;
using NeuralNetwork.Core.Optimizers;
using NeuralNetwork.Core.CNN;
using NeuralNetwork.Core.CNN.Layers;
using NeuralNetwork.Core.CNN.Activations;
using NeuralNetwork.Core.MLP.Losses;
using NeuralNetwork.Testing.AlgorithmTests;

namespace NeuralNetwork.Testing
{
    internal class Program
    {
        readonly static TextReaderWordVector TextReader = new ("D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt");

        static void Main(string[] args)
        {
            TestBest();
        }

        static void TestBest()
        {
            var modelPath = "D:/Projects/ml_models/cyberbullying_model_best.json";

            CyberbullyingAlgorithm.TestCyberbullyingModel(TextReader, modelPath);
        }

        static void TrainModel()
        {
            var modelPath = "D:/Projects/ml_models/cyberbullying_model.json";

            CyberbullyingAlgorithm.TrainCyberbullyingModel(TextReader, modelPath);
            CyberbullyingAlgorithm.TestCyberbullyingModel(TextReader, modelPath);
        }

        static void CnnAlgorithm()
        {
            var textReader = new TextReaderWordVector("D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt");
            var ((X, y), (XVal, yVal)) = SentimentsDataset.PrepareSentimentsDataset("D:/Datasets/twitter_sentiments/twitter_sentiments.csv", textReader);

            var model = new ModelConvolutional(new LossCategoricalCrossentropy(), new OptimizerSGD(), new AccuracyClassification());

            model.CNNLayers.Add(new LayerConvolution((25, 25, 1), 2, 10));
            model.CNNLayers.Add(new ActivationSigmoidConvolutional(
                model.CNNLayers.Last().OutputDepth,
                model.CNNLayers.Last().OutputRows,
                model.CNNLayers.Last().OutputColumns));
            model.PrepareFlattenLayer();

            model.MLPLayers.Add(new LayerDense(model.FlattenLayer.OutputLength, 32));

            model.MLPLayers.Add(new ActivationReLU());

            model.MLPLayers.Add(new LayerDense(32, 3));

            model.MLPLayers.Add(new ActivationSigmoid());

            model.PrepareLayers();

            model.PrepareOutput();

            // Forward pass
            model.Train((X, y), epochs: 100, logFreq: 100);
        }

        static void TestAlgorithm()
        {
            //var wordvecModel = "D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt";
            //var textReader = new TextReaderWordVector(wordvecModel);

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
    }
}
