using System;
using static System.Math;

namespace Neuro
{
    public interface IActivationFunction
    {
        double Function(double x);
        double Derivative(double x);
    }

    [Serializable]
    public class BipolarSigmoidFunction : IActivationFunction
    {
        public double Alpha { get; set; } = 2d;

        public BipolarSigmoidFunction() { }
        public BipolarSigmoidFunction(double alpha) =>
            Alpha = alpha;

        public double Function(double x) =>
            1d / (1d + Exp(-Alpha * x));

        public double Derivative(double x)
        {
            var y = Function(x);

            return Alpha * y * (1d - y);
        }

    }

    [Serializable]
    public class SigmoidFunction : IActivationFunction
    {
        public double Function(double x) =>
            1d / (1d + Exp(-x));

        public double Derivative(double x)
        {
            var y = Function(x);

            return y * (1d - y);
        }
    }

}
