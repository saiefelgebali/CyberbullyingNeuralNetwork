using Accord.Math;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.Layers;

namespace NeuralNetwork.Test.Layers
{
    [TestClass]
    public class LayerDenseTest
    {
        [TestMethod]
        public void TestForward()
        {
            // Setup layer
            var inputs = new double[][]
            {
                new double[] { 1, 2, 3 },
                new double[] { 7, 8, 9 },
            };
            var weights = new double[][]
            {
                new double [] { 1.0, 1.5 },
                new double [] { 0.5, -0.2 },
                new double [] { 2.0, 0.8 },
            };
            var biases = new double[] { 1, 0 };

            var layer = new LayerDense(numInputs: 3, numNeurons: 2, weights, biases);

            // Make a forward pass
            layer.Forward(inputs);

            // Expected output
            var expectedOutput = new double[][] 
            { 
                new double[] { 9, 3.5 },
                new double[] { 30, 16.1 },
            };

            // Correct for both samples
            // Rouding is done to account for binary rounding error
            Assert.IsTrue(Utility.ArrayEquals(expectedOutput.Round(), layer.Output.Round()));
        }

        [TestMethod]
        public void TestBackward()
        {
            // Setup layer
            var inputs = new double[][]
            {
                new double[] { 1, 2 },
            };

            var weights = new double[][]
            {
                new double [] { 2 },
                new double [] { 3 },
            };
            var biases = new double[] { 3 };

            var layer = new LayerDense(numInputs: 2, numNeurons: 1, weights, biases);

            layer.Inputs = inputs;

            // Perform a backward pass
            var dValues = new double[][]
            {
                new double[] { 14 },
            };

            layer.Backward(dValues);

            // Expected values
            // For each expected differential, we apply the chain rule
            // by multiplying by dValues (in this case, 14).
            var expectedDWeights = new double[][] 
            { 
                new double[] { 14 * 1 },
                new double[] { 14 * 2 },
            };

            var expectedDBiases = new double[] { 14 };

            var expectedDInputs = new double[][]
            {
                new double[] { 14 * 2, 14 * 3 },
            };

            // Test DWeights, DBiases and DInputs
            Assert.IsTrue(Utility.ArrayEquals(expectedDWeights.Round(), layer.DWeights.Round()));
            Assert.IsTrue(Utility.ArrayEquals(expectedDBiases.Round(), layer.DBiases.Round()));
            Assert.IsTrue(Utility.ArrayEquals(expectedDInputs.Round(), layer.DInputs.Round()));
        }
    }
}