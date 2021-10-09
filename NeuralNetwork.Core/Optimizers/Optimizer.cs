using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.Optimizers
{
    public abstract class Optimizer
    {
        public double LearningRate { get; set; }
        public double CurrentLearningRate { get; set; }
        public int Iterations { get; set; }

        // Call once before any param updates
        public abstract void PreUpdateParams();

        // Param updates
        public abstract void UpdateParams(LayerDense layer);

        // Call once after any param updates
        public abstract void PostUpdateParams();
    }
}
