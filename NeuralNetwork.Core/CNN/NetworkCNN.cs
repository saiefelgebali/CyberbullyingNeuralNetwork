using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.CNN
{
    public abstract class NetworkCNN : NetworkItem<NetworkCNN>
    {
        public override NetworkCNN Next { get; set; }
        public override NetworkCNN Prev { get; set; }

        public double[][][][] Inputs { get; protected set; }
        public double[][][][] Output { get; protected set; }
        public double[][][][] DInputs { get; protected set; }
    }
}
