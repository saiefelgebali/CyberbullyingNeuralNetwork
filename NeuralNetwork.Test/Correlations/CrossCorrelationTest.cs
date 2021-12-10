using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.Correlations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Test.Correlations
{
    [TestClass]
    public class CrossCorrelationTest
    {
        // Test valid cross-correlations
        [TestMethod]
        public void ValidCrossCorrelation_Test()
        {
            // Setup
            var input = new double[][]
            {
                new double[] { 1, 6, 2 },
                new double[] { 5, 3, 1 },
                new double[] { 7, 0, 4 },
            };

            var kernel = new double[][]
            {
                new double[] { 1, 2 },
                new double[] { -1, 0 },
            };

            // Apply operation
            var output = CrossCorrelation.ValidCrossCorrelation(input, kernel);

            // Expected output
            var expected = new double[][]
            {
                new double[] { 8, 7 },
                new double[] { 4, 5 },
            };

            // Check result
            Assert.IsTrue(Utility.ArrayEquals(expected, output));
        }

        [TestMethod]
        public void ValidCrossCorrelation_Kernel3x3_Test()
        {
            // Setup
            var input = new double[][]
            {
                new double[] { 1, 6, 2 },
                new double[] { 5, 3, 1 },
                new double[] { 7, 0, 4 },
            };

            var kernel = new double[][]
            {
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
            };

            // Apply operation
            var output = CrossCorrelation.ValidCrossCorrelation(input, kernel);

            // Expected output
            var expected = new double[][] { new double[] { input.Sum() } };

            // Check result
            Assert.IsTrue(Utility.ArrayEquals(expected, output));
        }
        
        [TestMethod]
        public void ValidCrossCorrelation_NonSquareInput_Test()
        {
            // Setup
            var input = new double[][]
            {
                new double[] { 1, 6, 2, 1 },
                new double[] { 5, 3, 1, 1 },
                new double[] { 7, 0, 4, 1 },
            };

            var kernel = new double[][]
            {
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
            };

            // Apply operation
            var output = CrossCorrelation.ValidCrossCorrelation(input, kernel);

            // Expected output
            var expected = new double[][] { new double[] { 29, 19 } };

            // Check result
            Assert.IsTrue(Utility.ArrayEquals(expected, output));
        }
        
        // Test full cross-correlations
        [TestMethod]
        public void FullCrossCorrelation_Test()
        {
            // Setup
            var input = new double[][]
            {
                new double[] { 1, 6, 2 },
                new double[] { 5, 3, 1 },
                new double[] { 7, 0, 4 },
            };

            var kernel = new double[][]
            {
                new double[] { 1, 2 },
                new double[] { -1, 0 },
            };

            // Apply full cross correlation
            var output = CrossCorrelation.FullCrossCorrelation(input, kernel);

            // Expected output
            var expected = new double[][] 
            { 
                new double[] { 0, -1, -6, -2 },
                new double[] { 2, 8, 7, 1 },
                new double[] { 10, 4, 5, -3 },
                new double[] { 14, 7, 8, 4 },
            };

            // Check result
            Assert.IsTrue(Utility.ArrayEquals(expected, output));
        }
                
        [TestMethod]
        public void FullCrossCorrelation_Kernel3x3_Test()
        {
            // Setup
            var input = new double[][]
            {
                new double[] { 1, 6, 2 },
                new double[] { 5, 3, 1 },
                new double[] { 7, 0, 4 },
            };

            var kernel = new double[][]
            {
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
            };

            // Apply full cross correlation
            var output = CrossCorrelation.FullCrossCorrelation(input, kernel);

            // Expected output
            var expected = new double[][] 
            { 
                new double[] { 1, 7, 9, 8, 2 },
                new double[] { 6, 15, 18, 12, 3 },
                new double[] { 13, 22, 29, 16, 7 },
                new double[] { 12, 15, 20, 8, 5 },
                new double[] { 7, 7, 11, 4, 4 },
            };

            // Check result
            Assert.IsTrue(Utility.ArrayEquals(expected, output));
        }        
        
        [TestMethod]
        public void FullCrossCorrelation_NonSquareInput_Test()
        {
            // Setup
            var input = new double[][]
            {
                new double[] { 1, 1, 1, 1 },
                new double[] { 1, 1, 1, 1 },
                new double[] { 1, 1, 1, 1 },
            };

            var kernel = new double[][]
            {
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
                new double[] { 1, 1, 1 },
            };

            // Apply full cross correlation
            var output = CrossCorrelation.FullCrossCorrelation(input, kernel);

            // Expected output
            var expected = new double[][] 
            { 
                new double[] { 1, 2, 3, 3, 2, 1 },
                new double[] { 2, 4, 6, 6, 4, 2 },
                new double[] { 3, 6, 9, 9, 6, 3 },
                new double[] { 2, 4, 6, 6, 4, 2 },
                new double[] { 1, 2, 3, 3, 2, 1 },
            };

            // Check result
            Assert.IsTrue(Utility.ArrayEquals(expected, output));
        }


    }
}
