using System;

namespace Neuro
{
    public class SelfOrganizingMapTeacher : ITeacher
    {
        private readonly Network _network;
        private static readonly Random s_random = new Random();
        private readonly int _sqrNeurons;

        private readonly double _sigma, _alpha;

        public int Iteration { get; private set; } = 0;

        public SelfOrganizingMapTeacher(Network network)
        {
            _network = network;
            _sqrNeurons = (int)Math.Sqrt(_network[0].Neurons.Length);

            _sigma = DataVisualizing.Settings1.Default.Sigma;
            _alpha = DataVisualizing.Settings1.Default.Alpha;
        }

        public double RunEpoch(double[][] input, double[][] output)
        {
            Iteration++;
            var vector = input[s_random.Next(0, input.Length)];
            var minNode = -1;
            var min = double.MaxValue;
            for (var j = 0; j < _network[0].Neurons.Length; j++)
            {
                var node = _network[0][j];
                var sum = 0.0d;
                for (var i = 0; i < vector.Length; i++)
                    sum += (vector[i] - node[i]) * (vector[i] - node[i]);
                sum = Math.Sqrt(sum);

                if (sum < min)
                {
                    min = sum;
                    minNode = j;
                }
            }

            double Distance(double x1, double y1, double x2, double y2) =>
                Math.Sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));

            for (var j = 0; j < _network[0].Neurons.Length; j++)
            {
                var node = _network[0][j];
                for (var i = 0; i < node.Weights.Length; i++)
                {
                    var sigma = _sqrNeurons * Math.Sqrt(2d) * Math.Exp(_sigma * Iteration);
                    var alpha = Math.Exp(_alpha * Iteration);
                    if (sigma != 0 && alpha != 0)
                        node[i] += alpha * Math.Exp(-Distance(minNode % _sqrNeurons, minNode / _sqrNeurons, j % _sqrNeurons, j / _sqrNeurons) / (2d * sigma * sigma))
                            * (vector[i] - node[i]);
                }
            }

            var error = 0d;
            for (var i = 0; i < input.Length; i++)
            {
                var sum = 0.0d;
                for (var j = 0; j < input[i].Length; j++)
                    sum += (input[i][j] - _network[0][minNode][j]) * (input[i][j] - _network[0][minNode][j]);
                sum = Math.Sqrt(sum);

                error += sum;
            }
            return error / input.Length;
        }
    }
}