namespace NeuralNetwork.Core.Layers
{
    public class LayerDenseParams
    {
        public double[][] Weights { get; set; }
        public double[] Biases { get; set; }

        public LayerDenseParams(double[][] weights, double[] biases)
        {
            Weights = weights;
            Biases = biases;
        }

    }
}
