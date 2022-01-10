using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.MLP.Activations
{
    public abstract class Activation : LayerMLP
    {
        public abstract int[] Predictions();
    }
}
