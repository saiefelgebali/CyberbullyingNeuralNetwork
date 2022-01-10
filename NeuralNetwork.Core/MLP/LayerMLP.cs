using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.MLP
{
    public abstract class LayerMLP : NetworkMLP
    {
        public abstract void Forward(double[][] input, bool training = false);

        public abstract void Backward(double[][] dValues);
    }
}
