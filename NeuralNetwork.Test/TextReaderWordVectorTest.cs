using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Core.Text;

namespace NeuralNetwork.Test
{
    [TestClass]
    public class TextReaderWordVectorTest
    {
        [TestMethod]
        public void TextReader_AverageWordVectors()
        {
            var vectors = new double[][]
            {
                new double[] { 0, 0, 4 },
                new double[] { 0, 3, 1 },
                new double[] { 0, 0, 1 },
            };

            var expected = new double[] { 0, 1, 2 };

            var actual = TextReaderWordVector.AverageWordVectors(vectors);

            Assert.IsTrue(Utility.ArrayEquals(actual, expected));
        }
    }
}
