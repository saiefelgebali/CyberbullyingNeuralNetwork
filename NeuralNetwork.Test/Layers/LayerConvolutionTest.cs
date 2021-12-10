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
            // - Takes in a 3x3 input
            // - Uses 2, 2x2 kernels
            var layer = new LayerConvolution((3, 3), 2, 2);

            var input = new double[][][]
            {
                new double[][]
                {
                    new double[] { 1, 6, 2 },
                    new double[] { 5, 3, 1 },
                    new double[] { 7, 0, 4 },
                }
            };

            layer.Forward(input);

            var output = layer.Output;

            Assert.IsTrue(false);
        }
    }
}
