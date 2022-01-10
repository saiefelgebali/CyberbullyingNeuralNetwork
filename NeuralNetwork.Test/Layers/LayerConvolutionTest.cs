using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Core.CNN.Layers;
using Accord.Math;

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
                    new double[] { 1, 2 },
                    new double[] { 1, 1 },
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
                        new double[] { 9, 9 },
                        new double[] { 5, 6 },
                    }
                },
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 9, 9 },
                        new double[] { 5, 6 },
                    }
                },
            };

            Assert.IsTrue(Utility.ArrayEquals(expected, layer.Output));
        }

        [TestMethod]
        public void Backward_Test()
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
                    new double[] { 1, 2 },
                    new double[] { 1, 1 },
                }
            };
            
            var layer = new LayerConvolution((3, 3, 1), kernels, biases);

            // Backprop
            var dValues = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 9, 9 },
                        new double[] { 5, 6 },
                    }
                }
            };

            layer.Forward(input);
            layer.Backward(dValues);

            // Expected derivatives
            var expectedDKernels = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 106, 93 },
                        new double[] { 107, 60 },
                    }
                }
            };

            var expectedDBiases = dValues.Flatten();

            var expectedDInputs = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 6, 17, 10 },
                        new double[] { 3, 22, 18 },
                        new double[] { -9, -9, 0 },
                    }
                }
            };

            Assert.IsTrue(Utility.ArrayEquals(expectedDKernels, layer.DKernels));
            Assert.IsTrue(Utility.ArrayEquals(expectedDBiases, layer.DBiases));
            Assert.IsTrue(Utility.ArrayEquals(expectedDInputs, layer.DInputs));
        }
    }
}
