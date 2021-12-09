using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Test.Layers
{
    [TestClass]
    public class LayerDropoutTest
    {
        [TestMethod]
        public void TestForward()
        {
            // Setup layer
            var inputs = new double[][]
            {
                new double[] { 10, 5 },
            };

            var binaryMask = new int[][]
            {
                new int[] { 0, 1 },
            };

            var layer = new LayerDropout(binaryMask);

            // Make a forward pass
            layer.Forward(inputs, training: true);

            // Expected output: first input is dropped
            var expectedOutput = new double[][]
            {
                new double[] { 0, 5 },
            };

            // Test dropout
            Assert.IsTrue(Utility.ArrayEquals(expectedOutput, layer.Output));
        }

        [TestMethod]
        public void TestBackward()
        {
            // Setup layer
            var binaryMask = new int[][]
            {
                new int[] { 0, 1 },
            };

            var layer = new LayerDropout(binaryMask);

            // Perform a backward pass
            var dValues = new double[][]
            {
                new double[] { 10, 5 },
            };

            layer.Backward(dValues);

            // Expected output: multiply by binary mask
            var expectedOutput = new double[][]
            {
                new double[] { 0, 5 },
            };

            // Test dropout
            Assert.IsTrue(Utility.ArrayEquals(expectedOutput, layer.DInputs));
        }
    }
}
