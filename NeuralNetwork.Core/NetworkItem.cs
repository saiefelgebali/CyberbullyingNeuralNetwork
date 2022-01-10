using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    public abstract class NetworkItem<T>
    {
        // Previous layer in network
        public abstract T Prev { get; set; }

        // Next layer in network
        public abstract T Next { get; set; }
    }
}
