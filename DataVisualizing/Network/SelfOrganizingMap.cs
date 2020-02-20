
namespace Neuro
{
    public class StraightActivation : IActivationFunction
    {
        public double Function(double x) =>
            x;
        public double Derivative(double x) =>
            1;
    }

    public class SelfOrganizingMap : Network
    {
        public SelfOrganizingMap(int inputsCount, int neuronsCount) : base(new StraightActivation(), inputsCount, neuronsCount * neuronsCount)
        {

        }

    }
}