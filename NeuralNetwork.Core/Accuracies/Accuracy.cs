using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using Accord.Statistics;

namespace NeuralNetwork.Core.Accuracies
{
    public abstract class Accuracy
    {
        public double Calculate(int[] yPred, int[] yTrue)
        {
            // Get comparison results
            int[] comparisons = Compare(yPred, yTrue);

            // Get accuracy using mean of comparisons
            double accuracy = Measures.Mean(comparisons);

            return accuracy;
        }

        public abstract int[] Compare(int[] yPred, int[] yTrue);
    }
}
