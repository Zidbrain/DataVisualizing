// implement momentum ??

using System;

namespace Neuro
{
    [Serializable]
    public class Backpropagation : ITeacher
    {
        private double[][] _deltaAct;
        private double[][][] _deltaWeight;
        private double[][] _deltaBias;

        public Network Network { get; private set; }

        public double LearningRate { get; set; }

        public Backpropagation(Network network)
        {
            Network = network;

            LearningRate = 0.1d;

            CreateDetlas();
        }

        private double CalculateCost(double[] input, double[] output)
        {
            var outp = Network.Compute(input);

            if (outp.Length != output.Length)
                throw new ArgumentException();

            var sum = 0d;

            for (var i = 0; i < outp.Length; i++)
            {
                var p = outp[i] - output[i];
                p *= p;
                sum += p;
            }

            return sum / 2d;
        }

        private void CreateDetlas()
        {
            _deltaAct = new double[Network.Layers.Length - 1][];
            _deltaWeight = new double[Network.Layers.Length][][];
            _deltaBias = new double[Network.Layers.Length][];

            for (var li = 0; li < Network.Layers.Length; li++)
            {
                var layer = Network[li];

                if (li <= Network.Layers.Length - 2)
                    _deltaAct[li] = new double[layer.Neurons.Length];
                _deltaBias[li] = new double[layer.Neurons.Length];
                _deltaWeight[li] = new double[layer.Neurons.Length][];

                for (var ni = 0; ni < layer.Neurons.Length; ni++)
                {
                    _deltaWeight[li][ni] = new double[Network[li][ni].Weights.Length];
                }
            }
        }

        public double RunEpoch(double[][] input, double[][] output, int start, int end)
        {
            if (input.Length != output.Length)
                throw new ArgumentException();

            var errorsum = 0d;

            for (var epoch = start; epoch < input.Length && epoch < start + end; epoch++)
            {
                errorsum += CalculateCost(input[epoch], output[epoch]);

                CreateDetlas();

                // last layer
                var lastL = Network.Layers.Length - 1;

                Layer layer = Network.Layers[lastL];

                for (var neuroni = 0; neuroni < layer.Neurons.Length; neuroni++)
                {
                    var neuron = layer[neuroni];

                    // calculate delta bias
                    _deltaBias[lastL][neuroni] += neuron.Function.Derivative(neuron.WeightedSum) * (neuron.Output - output[epoch][neuroni]);

                    // calculate delta weights
                    for (var wi = 0; wi < neuron.Weights.Length; wi++)
                    {
                        double inputActivation;

                        if (lastL == 0)
                            inputActivation = input[epoch][wi];
                        else
                            inputActivation = Network[lastL - 1][wi].Output;

                        _deltaWeight[lastL][neuroni][wi] += neuron.Function.Derivative(neuron.WeightedSum) * (neuron.Output - output[epoch][neuroni]) * inputActivation;
                    }
                }

                // backpropagate
                for (var li = lastL - 1; li >= 0; li--)
                {
                    layer = Network[li];

                    for (var ni = 0; ni < layer.Neurons.Length; ni++)
                    {
                        var neuron = layer[ni];

                        _deltaAct[li][ni] = CalculateDeltaAct(li, ni, output[epoch]);

                        _deltaBias[li][ni] += neuron.Function.Derivative(neuron.WeightedSum) * _deltaAct[li][ni];

                        for (var wi = 0; wi < neuron.Weights.Length; wi++)
                        {
                            var mult = li == 0 ? input[epoch][wi] : Network[li - 1][wi].Output;

                            _deltaWeight[li][ni][wi] += neuron.Function.Derivative(neuron.WeightedSum) * _deltaAct[li][ni] * mult;
                        }
                    }
                }
                Assign(1);
            }

            return errorsum / input.Length;
        }

        public double RunEpoch(double[][] input, double[][] output) =>
            RunEpoch(input, output, 0, input.Length);

        private void Assign(int lenght)
        {
            for (var li = 0; li < Network.Layers.Length; li++)
            {
                for (var ni = 0; ni < Network[li].Neurons.Length; ni++)
                {
                    var sq = (_deltaBias[li][ni] / lenght);
                    Network[li][ni].Bias -= sq * LearningRate;

                    for (var wi = 0; wi < Network[li][ni].Weights.Length; wi++)
                    {
                        var sq1 = _deltaWeight[li][ni][wi] / lenght;
                        Network[li][ni][wi] -= sq1 * LearningRate;
                    }
                }
            }
        }

        private double CalculateDeltaAct(int layer, int neuroni, double[] desired)
        {
            var sum = 0d;

            for (var ni = 0; ni < Network[layer + 1].Neurons.Length; ni++)
            {
                var neuronn = Network[layer + 1][ni];
                //not quite sure about that
                var mult = layer == Network.Layers.Length - 2 ? (neuronn.Output - desired[ni]) : _deltaAct[layer + 1][ni];

                sum += neuronn[neuroni] * neuronn.Function.Derivative(neuronn.WeightedSum) * mult;
            }

            return sum;
        }
    }
}
