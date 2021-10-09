using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    public class NetworkItem
    {
        // Network properties
        public double[][] Inputs { get; set; }
        public double[][] Output { get; set; }
        public double[][] DInputs {  get; set; }

        // Previous layer in network
        public NetworkItem Prev { get; set; }

        // Next layer in network
        public NetworkItem Next { get; set; }
    }
}
