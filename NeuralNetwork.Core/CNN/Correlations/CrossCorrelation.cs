using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Core.Correlations
{
    public static class CrossCorrelation
    {
        /// <summary>
        /// Returns a valid cross-correlation between two matrices.
        /// - Size of output O = I - K + 1
        /// </summary>
        public static double[][] ValidCrossCorrelation(double[][] input, double[][] kernel)
        {
            // Ensure kernel is a square matrix
            if (kernel.Rows() != kernel.Columns())
            {
                throw new ArgumentException("Kernel must be a square matrix");
            }

            // Setup output matrix
            var kernelSize = kernel.Rows();
            var outputRows = input.Rows() - kernelSize + 1;
            var outputColumns = input.Columns() - kernelSize + 1;
            var output = Matrix.Create<double>(outputRows, outputColumns).ToJagged();

            // Loop through different kernel positions
            for (int i = 0; i < output.Rows(); i++)
            {
                for (int j = 0; j < output.Columns(); j++)
                {
                    // Take a slice from input matrix
                    // at current position
                    // with size of kernel
                    var slice = SliceMatrix(input, i, j, kernelSize);
                    output[i][j] = slice.Multiply(kernel).Sum();
                }
            }

            return output;
        }

        private static double[][] SliceMatrix(double[][] input, int startRow, int startColumn, int size)
        {
            var slice = Matrix.Create<double>(size, size).ToJagged();

            // Cut out a square matrix from existing matrix
            // starting from the top-left of a certain point
            for (int row = 0; row < size; row++)
            {
                for (int col = 0; col < size; col++)
                {
                    slice[row][col] = input[startRow + row][startColumn + col];
                }
            }

            return slice;
        }

        /// <summary>
        /// Returns a full cross-correlation between two matrices.
        /// - Size of output O = I + K - 1
        /// </summary>
        public static double[][] FullCrossCorrelation(double[][] input, double[][] kernel)
        {
            // Ensure kernel is a square matrix
            if (kernel.Rows() != kernel.Columns())
            {
                throw new ArgumentException("Kernel must be a square matrix");
            }

            // Setup output matrix
            var kernelSize = kernel.Rows();
            var outputRows = input.Rows() + kernelSize - 1;
            var outputColumns = input.Columns() + kernelSize - 1;
            var output = Matrix.Create<double>(outputRows, outputColumns).ToJagged();

            // Loop through different kernel positions
            for (int i = 0; i < output.Rows(); i++)
            {
                for (int j = 0; j < output.Columns(); j++)
                {
                    // Take a slice from input matrix
                    // at current position
                    // with size of kernel
                    var slice = SliceMatrixFull(input, i, j, kernelSize);
                    output[i][j] = slice.Multiply(kernel).Sum();
                }
            }

            return output;
        }

        private static double[][] SliceMatrixFull(double[][] input, int currentRow, int currentColumn, int size)
        {
            var slice = Matrix.Create<double>(size, size).ToJagged();

            // Cut out a square matrix from existing matrix
            // starting from the bottom-right of a certain point.
            // Inputs outside of edge default to 0
            for (int row = 0; row < size; row++)
            {
                for (int col = 0; col < size; col++)
                {
                    var inputRow = currentRow - row;
                    var inputColumn = currentColumn - col;

                    double inputValue;

                    // Check if current input is outside of matrix,
                    // default to 0
                    if (inputRow >= input.Rows() || inputRow < 0)
                    {
                        inputValue = 0;
                    }
                    else if (inputColumn >= input.Columns() || inputColumn < 0)
                    {
                        inputValue = 0;
                    }
                    // Otherwise use input value
                    else
                    {
                        inputValue = input.ElementAt(inputRow).ElementAt(inputColumn);
                    }

                    slice[size - row - 1][size - col - 1] = inputValue;
                }
            }

            return slice;
        }


        /// <summary>
        /// Rotate a matrix by 180 degrees
        /// - 1: Transpose matrix
        /// - 2: Reverse columns
        /// - 3: Transpose matrix
        /// - 4: Reverse columns
        /// </summary>
        private static double[][] Rotate180(double[][] matrix)
        {
            var currentMatrix = matrix.Copy();

            // Repeat operations twice
            for (int i = 0; i < 2; i++)
            {
                // Transpose
                currentMatrix = currentMatrix.Transpose();

                // Reverse columns
                for (int row = 0; row < currentMatrix.Rows(); row++)
                {
                    currentMatrix[row] = currentMatrix[row].Reverse().ToArray();
                }
            }

            return currentMatrix;
        }

        public static double[][] ValidConvolultion(double[][] input, double[][] kernel)
        {
            return Rotate180(ValidCrossCorrelation(input, kernel));
        }

        public static double[][] FullConvolultion(double[][] input, double[][] kernel)
        {
            return Rotate180(FullCrossCorrelation(input, kernel));
        }
    }
}
