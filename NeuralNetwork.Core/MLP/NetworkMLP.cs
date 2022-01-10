using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.MLP
{
    public abstract class NetworkMLP : NetworkItem<NetworkMLP>
    {
        public override NetworkMLP Next { get; set; }
        public override NetworkMLP Prev { get; set; }

        public double[][] Inputs { get; set; }
        public double[][] Output { get; protected set; }
        public double[][] DInputs { get; set; }
    }
}
