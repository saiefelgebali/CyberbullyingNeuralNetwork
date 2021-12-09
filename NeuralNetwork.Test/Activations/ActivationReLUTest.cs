using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.Activations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Test.Activations
{
    [TestClass]
    public class ActivationReLUTest
    {
        [TestMethod]
        public void Forward()
        {
            // Setup layer
            var input = new double[][]
            {
                new double[] { 1, 2, -3 },
                new double[] { 4, -5, 6 },
            };

            var layer = new ActivationReLU();

            // Perform a forward pass
            layer.Forward(input);

            // Expected output
            var expectedOutput = new double[][]
            {
                new double[] { 1, 2, 0 },
                new double[] { 4, 0, 6 },
            };

            Assert.IsTrue(Utility.ArrayEquals(expectedOutput, layer.Output));
        }

        [TestMethod]
        public void Backward()
        {
            // Setup layer
            var input = new double[][]
            {
                new double[] { 2, -1, 0 },
                new double[] { 1, 2, -3 },
            };

            var layer = new ActivationReLU();

            layer.Inputs = input;

            // Perform a backward pass
            var dValues = new double[][]
            {
                new double[] { 4, -5, -6},
                new double[] { -7, 8, 9 },
            };

            layer.Backward(dValues);

            // Expected result
            var expectedOutput = new double[][]
            {
                new double[] { 4, 0, -6 },
                new double[] { -7, 8, 0 },
            };

            Assert.IsTrue(Utility.ArrayEquals(expectedOutput, layer.DInputs));
        }
    }
}
