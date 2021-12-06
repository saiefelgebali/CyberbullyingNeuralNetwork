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
using NeuralNetwork.Testing;

namespace NeuralNetwork.Testing
{
    internal class Program
    {
        static void Main(string[] args)
        {
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
