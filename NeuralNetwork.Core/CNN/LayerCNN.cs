using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.CNN
{
    public abstract class LayerCNN : NetworkCNN
    {
        public abstract int OutputDepth { get; protected set; }
        public abstract int OutputRows { get; protected set; }
        public abstract int OutputColumns { get; protected set; }

        public abstract void Forward(double[][][][] input);
        public abstract void Backward(double[][][][] input);
    }
}
