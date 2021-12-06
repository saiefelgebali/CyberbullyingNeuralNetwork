using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.Layers;
using System.Linq;

namespace NeuralNetwork.Test
{
    [TestClass]
    public class TestLayerDense
    {
        [TestMethod]
        public void TestForwardSingle()
        {
            // Setup layer
            var inputs = new double[][] { new double[] { 1, 1, 1, 1 } };
            var weights = new double[][] { new double [] { 64, 64, 64, 64 } };
            var biases = new double[] { 64, 64, 64, 64 };

            var layer = new LayerDense(weights, biases);

            // Make a forward pass
            layer.Forward(inputs);

            // Expected output
            var expected = new double[][] { new double[] { 128, 128, 128, 128 } };

            // Correct output length
            Assert.AreEqual(layer.Output.Length - expected.Length, 0);

            // Correct output
            Assert.IsTrue(expected.First().SequenceEqual(layer.Output.First()));
        }

        [TestMethod]
        public void TestForwardBatch()
        {
            // Setup layer
            var inputs = new double[][] 
            { 
                new double[] { 1, 2 },
                new double[] { 4, 8 }
            };
            var weights = new double[][] 
            { 
                new double[] { 64, 64, 64, 64 }, 
                new double[] { 64, 64, 64, 64 }, 
            };
            var biases = new double[] { 64, 64, 64, 64 };

            var layer = new LayerDense(weights, biases);

            // Make a forward pass
            layer.Forward(inputs);

            // Expected output
            var expected = new double[][] 
            { 
                new double[] { 128, 128, 128, 128 },
                new double[] { 128, 128, 128, 128 },
            };

            // Correct output length
            Assert.AreEqual(layer.Output.Length - expected.Length, 0);

            // Correct output
            Assert.IsTrue(expected.First().SequenceEqual(layer.Output.First()));
        }
    }
}