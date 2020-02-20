using System;

namespace Neuro
{
    [Serializable]
    public class Layer
    {
        public int InputsCount { get; private set; }
        public Neuron[] Neurons { get; private set; }
        public double[] Output { get; private set; }

        public Neuron this[int index] => Neurons[index];

        private IActivationFunction _function;
        public IActivationFunction Function
        {
            get => _function;
            set
            {
                foreach (var neuron in Neurons)
                {
                    neuron.Function = value;
                }
                _function = value;
            }
        }

        public Layer(IActivationFunction func, int neuronsCount, int inputsCount)
        {
            InputsCount = inputsCount;
            Neurons = new Neuron[neuronsCount];

            for (var i = 0; i < Neurons.Length; i++)
                Neurons[i] = new Neuron(func, inputsCount);

            _function = func;
        }

        public Layer(int neuronsCount, int inputsCount) : this(new BipolarSigmoidFunction(), neuronsCount, inputsCount) { }

        public double[] Compute(double[] input)
        {
            var output = new double[Neurons.Length];

            for (var i = 0; i < Neurons.Length; i++)
                output[i] = Neurons[i].Compute(input);

            Output = output;

            return output;
        }

        public void Randomize()
        {
            foreach (var neuron in Neurons)
                neuron.Randomize();
        }
    }
}
