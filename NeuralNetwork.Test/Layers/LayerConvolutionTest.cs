using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Core.Layers;

namespace NeuralNetwork.Test.Layers
{
    [TestClass]
    public class LayerConvolutionTest
    {
        [TestMethod]
        public void Forward_Test()
        {
            // Setup layer
            var input = new double[][][][]
            {
                new double[][][]
                {
                    new double[][] 
                    {
                        new double[] { 1, 6, 2 },
                        new double[] { 5, 3, 1 },
                        new double[] { 7, 0, 4 },
                    },
                },
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 1, 6, 2 },
                        new double[] { 5, 3, 1 },
                        new double[] { 7, 0, 4 },
                    },
                }
            };

            var kernels = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 1, 2 },
                        new double[] { -1, 0 }
                    }
                }
            };

            var biases = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0, 0 },
                    new double[] { 0, 0 },
                }
            };

            var layer = new LayerConvolution((3,3,1), kernels, biases);

            // Perform forward pass
            layer.Forward(input);

            // Expected output
            var expected = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 8, 7 },
                        new double[] { 4, 5 },
                    }
                },
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 8, 7 },
                        new double[] { 4, 5 },
                    }
                },
            };

            Assert.IsTrue(Utility.ArrayEquals(expected, layer.Output));
        }
    }
}
