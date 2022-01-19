using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Prototype
{
    [TestClass]
    public class LayerDenseTest
    {
        private readonly LayerDense Layer;

        public LayerDenseTest()
        {
            // initialize weights parameter of layer
            var weights = new double[,]
            {
                { 1, 2 },
                { 2, 3 },
                { 3, 4 },
            };

            // initialize layer
            Layer = new LayerDense(numInputs: 3, numOutputs: 2, initialWeights: weights);
        }

        private static bool ArraysAreEqual()
        {

        }

        [TestMethod]
        public void TestForward()
        {
            // input to the layer
            var X = new double[,]
            {
                { 1, 2, 3 },
                { -1, -2, -3 },
            };

            // perform forward pass
            var output = Layer.Forward(X);

            // expected result to make valid comparison
            var expected = new double[,]
            {
                { 14, 20 },
                { -14, -20 },
            };

            // test if output matches expected matrix
            Assert.IsNotNull(output);
            for (int i = 0; i < output.Rows(); i++)
            {
                for (int j = 0; j < output.Columns(); j++)
                {
                    Assert.AreEqual(output[i, j], expected[i, j]);
                }
            }
        }

        [TestMethod]
        public void TestBackward()
        {
            // mock derivative values from next function
            var dValues = new double[,]
            {
                { -2, -2 },
                { 2, 4 },
            };

            Layer.Backward(dValues);

            var expectedDWeights = new double[,]
            {
                { 0, 0, 0 },
                { 6, 12, 14 },
            };

            var expectedDBiases = new double[] { -4, 4 };

            var expectedDInputs = new double[,]
            {
                { -6, -10, -14 },
                { -6, -8, -10 }
            };

            Layer.Backward(dValues);

            var dInputs = Layer.DInputs;
            var dWeights = Layer.DWeights;
            var dBiases = Layer.DBiases;

            for (int i = 0; i < dInputs.Rows(); i++)
            {
                for (int i = 0; i < dInputs.Columns(); i++)
                {

                }
            }
        }
    }
}
