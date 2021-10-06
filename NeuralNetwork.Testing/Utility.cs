using Accord.Math;
using Accord.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    public class Utility
    {
        // Console log a matrix
        public static void ShowMatrix(double[,] matrix)
        {
            for (int i = 0; i < matrix.Rows(); i++)
            {
                for (int j = 0; j < matrix.Columns(); j++)
                {
                    string value = matrix[i, j].ToString();
                    Console.Write(value.Substring(0, Math.Min(7, value.Length)));
                    Console.Write(" ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}
