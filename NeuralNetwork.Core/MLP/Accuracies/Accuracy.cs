using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using Accord.Statistics;

namespace NeuralNetwork.Core.MLP.Accuracies
{
    public abstract class Accuracy
    {
        public double AccumulatedSum { get; private set; }
        public double AccumulatedCount { get; private set; }

        public double Calculate(int[] yPred, int[] yTrue)
        {
            // Get comparison results
            int[] comparisons = Compare(yPred, yTrue);

            // Get accuracy using mean of comparisons
            double accuracy = Measures.Mean(comparisons);

            // Add accumulated sum and count
            AccumulatedSum += comparisons.Sum();
            AccumulatedCount += comparisons.Length;

            return accuracy;
        }

        public double CalculateAccumulated()
        {
            // Calculate accumulated mean
            double accuracy = AccumulatedSum / AccumulatedCount;

            return accuracy;
        }

        public void NewPass()
        {
            // Reset accumulated
            AccumulatedSum = 0;
            AccumulatedCount = 0;
        }

        public abstract int[] Compare(int[] yPred, int[] yTrue);
    }
}
