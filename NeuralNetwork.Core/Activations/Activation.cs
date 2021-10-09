using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.Activations
{
    public abstract class Activation : NetworkLayer
    {
        public abstract int[] Predictions();
    }
}
