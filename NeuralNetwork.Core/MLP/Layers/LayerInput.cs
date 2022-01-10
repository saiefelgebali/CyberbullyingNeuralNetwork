using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.MLP.Layers
{
    public class LayerInput : LayerMLP
    {
        public override void Forward(double[][] input, bool training = false)
        {
            Output = Inputs = input;
        }

        public override void Backward(double[][] dValues)
        {
            throw new NotImplementedException();
        }
    }
}
