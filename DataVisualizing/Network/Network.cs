using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace Neuro
{
    public interface ITeacher
    {
        double RunEpoch(double[][] input, double[][] output);
    }

    [Serializable]
    public class Network
    {
        public int InputsCount { get; private set; }
        public Layer[] Layers { get; private set; }
        public double[] Outputs { get; private set; }
        public double Output { get; private set; }

        private IActivationFunction _function;
        public IActivationFunction Function
        {
            get => _function;
            set
            {
                foreach (var layer in Layers)
                {
                    layer.Function = value;
                }
                _function = value;
            }
        }

        public Network Copy()
        {
            var neuronsCount = new int[Layers.Length];
            for (int i = 0; i < Layers.Length; i++)
                neuronsCount[i] = Layers[i].Neurons.Length;

            var ret = new Network(Function, InputsCount, neuronsCount);
            for (int i =0; i < ret.Layers.Length; i++)
                for (int j = 0; j < ret[i].Neurons.Length; j++)
                {
                    ret[i][j].Bias = this[i][j].Bias;
                    for (int k = 0; k < ret[i][j].Weights.Length; k++)
                        ret[i][j][k] = this[i][j][k];
                }
            return ret;
        }

        public Layer this[int index] => Layers[index];

        public Network(IActivationFunction func, int inputsCount, params int[] neuronsCount)
        {
            InputsCount = inputsCount;

            Layers = new Layer[neuronsCount.Length];

            for (var i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(func,
                    neuronsCount[i],
                    (i == 0) ? inputsCount : neuronsCount[i - 1]);
            }

            _function = func;
        }

        public Network(int inputsCount, params int[] neuronsCount) : this(new BipolarSigmoidFunction(), inputsCount, neuronsCount) { }

        public double[] Compute(double[] input)
        {
            var output = input;

            for (var i = 0; i < Layers.Length; i++)
            {
                output = Layers[i].Compute(output);
            }

            Outputs = output;

            return output;
        }

        public double ComputePropose(double[] inputs, out int propositionIndex)
        {
            var outputs = Compute(inputs);

            var max = 0d;
            propositionIndex = 0;

            for (var i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] > max)
                {
                    max = outputs[i];
                    propositionIndex = i;
                }
            }

            return max;
        }

        public int ComputePropose(double[] inputs)
        {
            ComputePropose(inputs, out var nil);
            return nil;
        }

        public void Randomize()
        {
            foreach (var layer in Layers)
            {
                layer.Randomize();
            }
        }

        public void Save(string fileName)
        {
            var stream = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.None);
            Save(stream);
            stream.Close();
        }

        public void Save(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, this);
        }

        public static Network Load(string fileName)
        {
            var stream = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            var network = Load(stream);
            stream.Close();

            return network;
        }

        public static Network Load(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            var network = (Network)formatter.Deserialize(stream);
            return network;
        }
    }
}
