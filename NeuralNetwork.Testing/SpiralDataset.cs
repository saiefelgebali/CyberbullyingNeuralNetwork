using Accord.Math;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Testing
{
    public class SpiralDataset
    {
        //define dataset
        public static (double[][] X, int[] y) GenerateSpiralData(int points, int classes)
        {
            var M = Matrix<double>.Build; //shortcut to Matrix builder
            var V = Vector<double>.Build; //shortcut to Vector builder

            //build vectors of size points*classesx1 for y, r and theta
            var Y = V.Dense(points * classes); //at this point this is full of zeros
            for (int j = 0; j < classes; j++)
            {
                var y_step = V.DenseOfArray(Generate.Step(points * classes, 1, (j + 1) * points));
                Y = Y + y_step;
            }
            var r = V.DenseOfArray(Generate.Sawtooth(points * classes, points, 0, 1));
            var theta = 4 * (r + Y) + (V.DenseOfArray(Generate.Standard(points * classes)) * 0.2);
            var sin_theta = theta.PointwiseSin();
            var cos_theta = theta.PointwiseCos();


            double[][] X = M.DenseOfColumnVectors(r.PointwiseMultiply(sin_theta), r.PointwiseMultiply(cos_theta)).ToArray().ToJagged();

            // convert y values to ints, and use one-hot vectors
            int[] y = Y.Select((val) => (int)val).ToArray();

            return (X, y);
        }
    }
}
