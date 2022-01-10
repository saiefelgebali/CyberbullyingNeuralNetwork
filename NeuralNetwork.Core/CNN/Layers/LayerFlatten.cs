using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Core.CNN.Layers
{
    public class LayerFlatten
    {
        // Dimensions
        public int InputRows { get; private set; }
        public int InputColumns { get; private set; }
        public int InputDepth { get; private set; }
        public int OutputLength { get; private set; }
        // Networking
        public double[][][][] Inputs { get; private set; }
        public double[][] Output { get; private set; }
        public double[][][][] DInputs { get; private set; }

        public LayerFlatten(int depth, int rows, int columns)       
        {
            InputRows = rows;
            InputColumns = columns;
            InputDepth = depth;
            OutputLength = InputDepth * InputRows * InputColumns;
        }

        public void Forward(double[][][][] input)
        {
            Inputs = input;

            Output = input.Select((s) => s.Flatten().Flatten()).ToArray();
        }

        public void Backward(double[][] dValues)
        {
            DInputs = new double[dValues.Length][][][];
            for (int sample = 0; sample < dValues.Length; sample++)
            {
                DInputs[sample] = new double[InputDepth][][];
                for (int depth = 0; depth < InputDepth; depth++)
                {
                    DInputs[sample][depth] = dValues[sample]
                        .Skip(depth * InputColumns * InputRows)
                        .Take(InputColumns * InputRows).ToArray()
                        .Reshape(InputRows, InputColumns).ToJagged();
                }
            }
        }
    }
}
