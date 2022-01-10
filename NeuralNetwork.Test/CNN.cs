using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.MLP;
using NeuralNetwork.Core.MLP.Layers;
using NeuralNetwork.Core.MLP.ActivationLoss;
using NeuralNetwork.Core.MLP.Accuracies;
using NeuralNetwork.Core.MLP.Activations;
using NeuralNetwork.Core.Optimizers;
using NeuralNetwork.Core.CNN;
using NeuralNetwork.Core.CNN.Layers;
using NeuralNetwork.Core.CNN.Activations;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.MLP.Losses;

namespace NeuralNetwork.Test
{
    [TestClass]
    public class CNN
    {
        [TestMethod]
        public void CNN_Test()
        {
            var X = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 1, 2000 },
                        new double[] { -321, 4 },
                    }
                }
            };

            var model = new ModelConvolutional(new LossCategoricalCrossentropy(), new OptimizerSGD(), new AccuracyClassification());

            model.CNNLayers.Add(new LayerConvolution((2, 2, 1), 2, 10));
            model.CNNLayers.Add(new ActivationSigmoidConvolutional(
                model.CNNLayers.Last().OutputDepth,
                model.CNNLayers.Last().OutputRows, 
                model.CNNLayers.Last().OutputColumns));
            model.PrepareFlattenLayer();

            model.MLPLayers.Add(new LayerDense(model.FlattenLayer.OutputLength, 32));

            model.MLPLayers.Add(new ActivationReLU());

            model.MLPLayers.Add(new LayerDense(32, 2));

            model.MLPLayers.Add(new ActivationSigmoid());

            model.PrepareLayers();

            model.PrepareOutput();

            // Forward pass
            var yTrue = new int[] { 1 };

            model.Train((X, yTrue));

            Assert.IsTrue(false);
        }
    }
}
