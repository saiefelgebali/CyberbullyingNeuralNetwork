using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.CNN.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Test.Layers
{
    [TestClass]
    public class LayerFlattenTest
    {
        [TestMethod]
        public void Forward_Test()
        {
            var input = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 1, 2 },
                        new double[] { 2, 3 },
                    },
                },
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 1, 2 },
                        new double[] { 2, 3 },
                    },
                }
            };

            var layer = new LayerFlatten(2,2,2);

            layer.Forward(input);

            // Expected result
            var expectedOutput = new double[][]
            {
                new double[] { 1, 2, 2, 3 },
                new double[] { 1, 2, 2, 3 },
            };

            Assert.IsTrue(Utility.ArrayEquals(expectedOutput, layer.Output));
        }

        [TestMethod]
        public void Backward_Test()
        {
            var dValues = new double[][]
            {
                new double[] { 1, 2, 2, 3, 1, 2, 2, 3 },
                new double[] { 4, 5, 6, 7, 7, 6, 5, 4 },
            };

            var layer = new LayerFlatten(2,2,2);

            layer.Backward(dValues);

            // Expected result
            var expectedDInputs = new double[][][][]
            {
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 1, 2 },
                        new double[] { 2, 3 },
                    },
                    new double[][]
                    {
                        new double[] { 1, 2 },
                        new double[] { 2, 3 },
                    },
                },
                new double[][][]
                {
                    new double[][]
                    {
                        new double[] { 4, 5 },
                        new double[] { 6, 7 },
                    },
                    new double[][]
                    {
                        new double[] { 7, 6 },
                        new double[] { 5, 4 },
                    },
                }
            };

            Assert.IsTrue(Utility.ArrayEquals(expectedDInputs, layer.DInputs));
        }
    }
}
