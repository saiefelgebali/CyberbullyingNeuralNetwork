using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Core.Accuracies
{
    public class AccuracyClassification : Accuracy
    {
        public override int[] Compare(int[] yPred, int[] yTrue)
        {
            // Loop over sample results and return whether or not classification was correct
            int[] correct = new int[yPred.Length];
            for (int i = 0; i < yPred.Length; i++)
            {
                bool compare = yPred[i] == yTrue[i];
                correct[i] = compare ? 1 : 0;
            }

            return correct;
        }
    }
}
