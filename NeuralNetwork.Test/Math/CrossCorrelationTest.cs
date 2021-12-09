﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Core.Correlations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Test.Math
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

        }
    }
}
