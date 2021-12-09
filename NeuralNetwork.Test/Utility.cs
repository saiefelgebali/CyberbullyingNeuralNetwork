using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Test
{
    internal class Utility
    {
        // Returns true if two 1D arrays of doubles are equal in value
        public static bool ArrayEquals(double[] arr1, double[] arr2)
        {
            if (arr1.Length != arr2.Length) return false;

            for (int i = 0; i < arr1.Length; i++)
            {
                if (arr1[i] != arr2[i]) return false;
            }

            return true;
        }

        // Returns true if two 2D arrays of doubles are equal in value
        public static bool ArrayEquals(double[][] arr1, double[][] arr2)
        {
            if (arr1.Length != arr2.Length) return false;

            for (int i = 0; i < arr1.Length; i++)
            {
                if (!ArrayEquals(arr1[i], arr2[i])) return false;
            }

            return true;
        }
    }
}
