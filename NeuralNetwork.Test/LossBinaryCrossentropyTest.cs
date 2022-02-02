using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using Accord.Statistics;

namespace NeuralNetwork.Test
{
    class LossBinaryCrossEntropy
    {
        public static double[] Forward(double[][] yPred, int[] _yTrue)
        {
            var yTrue = _yTrue.ToDouble().Transpose().ToJagged();
            var lossPositive = yTrue.Multiply(yPred.Log());
            var lossNegative = yTrue.Multiply(-1).Add(1).Multiply(yPred.Multiply(-1).Add(1).Log());
            return Measures.Mean(lossPositive.Add(lossNegative).Multiply(-1), dimension: 1);
        }
    }

    [TestClass]
    public class LossBinaryCrossentropyTest
    {
        [TestMethod]
        public void Forward()
        {
            var yPred = new double[][]
            {
                new double[] { 0, },
                new double[] { 0.5, },
                new double[] { 1, },
            };

            var yTrue = new int[] { 0, 0, 0 };

            var loss = LossBinaryCrossEntropy.Forward(yPred, yTrue);

            Assert.IsTrue(Math.Round(loss[0]) == 0);
            Assert.IsTrue(loss[2] > 0);
            Assert.IsTrue(loss[3] > loss[2]);
        }
    }
}
