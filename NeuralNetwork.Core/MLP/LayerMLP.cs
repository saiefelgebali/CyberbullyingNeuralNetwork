namespace NeuralNetwork.Core.MLP
{
    // This is an abstract class
    // It is used to define the properties and methods of its sub classes
    // However, this class is not meant to be instantiated on its own
    public abstract class LayerMLP : NetworkMLP
    {
        // An abstract method means that it is up to the sub-classes to implement it.
        public abstract void Forward(double[][] input, bool training = false);

        public abstract void Backward(double[][] dValues);
    }
}
