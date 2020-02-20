using System;

namespace Neuro
{
    [Serializable]
    public class Neuron
    {
        public int InputsCount { get; private set; }
        public double[] Weights { get; private set; }
        public double Output { get; private set; }
        public double WeightedSum { get; private set; }

        public double Bias { get; set; }

        public IActivationFunction Function { get; set; }

        public double this[int index]
        {
            get => Weights[index];
            set => Weights[index] = value;
        }

        private static readonly Random s_rand = new Random();

        public Neuron(IActivationFunction func, int inputs)
        {
            if (inputs < 1)
                throw new ArgumentException();

            InputsCount = inputs;
            Weights = new double[InputsCount];

            Function = func;

            Randomize();
        }

        public Neuron(int inputs) : this(new BipolarSigmoidFunction(), inputs) { }

        public void Randomize()
        {
            for (var i = 0; i < InputsCount; i++)
                Weights[i] = s_rand.NextDouble() * 2d - 1;
        }

        public double SumFunction(double[] input)
        {
            if (input.Length != InputsCount)
                throw new ArgumentException("Wrong length of the input vector.");

            var sum = 0.0;

            for (var i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * input[i];
            }
            sum += Bias;

            return sum;
        }

        public double Compute(double[] input)
        {
            var sum = SumFunction(input);

            WeightedSum = sum;

            var output = Function.Function(sum);
            Output = output;

            return output;
        }
    }
}
