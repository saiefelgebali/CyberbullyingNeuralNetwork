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
        public static void ShowMatrix<T>(T[][] matrix)
        {
            for (int i = 0; i < matrix.Rows(); i++)
            {
                for (int j = 0; j < matrix.Columns(); j++)
                {
                    string value = matrix[i][j].ToString();
                    Console.Write(value.Substring(0, Math.Min(7, value.Length)));
                    Console.Write(" ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        public static T[] ShuffleArray<T>(T[] array, int[] indices)
        {
            T[] shuffledArray = new T[array.Length];

            // Shuffle array based on new indices
            for (int i = 0; i < array.Length; i++)
            {
                shuffledArray[i] = array[indices[i]];
            }

            return shuffledArray;
        }

        public static int[] ShuffleIndices(int length)
        {
            int[] indices = new int[length];

            // Init indices with correct values
            for (int i = 0; i < indices.Length; i++)
            {
                indices[i] = i;
            }

            // Shuffle array
            indices.Shuffle();

            return indices;
        }
    }
}
