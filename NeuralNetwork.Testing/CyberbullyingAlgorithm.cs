using Accord.Math;
using NeuralNetwork.Core.Layers;
using NeuralNetwork.Core.Accuracies;
using NeuralNetwork.Core.Optimizers;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.Activations;
using NeuralNetwork.Core.Losses;
using NeuralNetwork.Core.Text;
using System;
using System.Linq;

namespace NeuralNetwork.Testing
{
    public class CyberbullyingAlgorithm
    {
        public static void TrainCyberbullyingModel(TextReaderWordVector textReader, string @savePath)
        {
            // Specify dataset path
            string datasetPath = "D:/Datasets/cyberbullying/cyberbullying_binary_dataset.csv";

            // Prepare dataset
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
            model.Train((X, y), (XVal, yVal), batchSize: 128, epochs: 1, logFreq: 1000);

            // Save model
            model.SaveParameters(@savePath);
        }
    }
}
