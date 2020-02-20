using System;
using System.IO;
using System.Collections.Generic;
using Neuro;
using System.Drawing;

namespace DataVisualizing
{
    public class DistanceActivation : IActivationFunction
    {
        public double Alpha { get; set; } = 0.5d;

        public double Function(double x) =>
            x * Alpha;

        public double Derivative(double x) =>
            Alpha;
    }

    public static class Program
    {
        private const int s_recSize = 60;
        private const int s_recDif = 2;
        private const int s_recBatchDif = 40;
        private const int s_recLineCount = 9;

        public static PointF[] FromCoordinates(int x, int y, int size)
        {
            size /= 2;
            return new PointF[]
             {
                new PointF(x, y - size),
                new PointF(x + size, y - 0.5f * size),
                new PointF(x + size, y + 0.5f * size),
                new PointF(x, y + size),
                new PointF(x - size, y + 0.5f * size),
                new PointF(x - size, y - 0.5f * size),
             };
        }

        public static double EuclidianDistance(Neuron neuron, double[] vector)
        {
            var sum = 0.0d;
            for (var i = 0; i < vector.Length; i++)
                sum += (neuron[i] - vector[i]) * (neuron[i] - vector[i]);
            return Math.Sqrt(sum);
        }


        public static Color Lerp(Color from, Color to, double value) =>
            Color.FromArgb((int)((to.A - from.A) * value + from.A), (int)((to.R - from.R) * value + from.R), (int)((to.G - from.G) * value + from.G), (int)((to.B - from.B) * value + from.B));

        public static Image Draw(Neuron[] neurons, double[][] inputs)
        {
            var inputsCount = neurons[0].Weights.Length;
            var image = new Bitmap((s_recSize + s_recDif) * s_recXCount + (s_recLineCount - 1) * ((s_recSize + s_recDif) * s_recXCount + s_recBatchDif) + s_recSize / 2,
                                    neurons.Length / s_recXCount * (s_recSize + s_recDif) + inputsCount / s_recLineCount * ((s_recSize + s_recDif) * neurons.Length / s_recXCount));
            var graphics = Graphics.FromImage(image);
            graphics.FillRectangle(Brushes.White, 0, 0, image.Size.Width, image.Size.Height);

            var font = new Font(SystemFonts.DefaultFont.FontFamily, 60f);

            void Draw(int inpCount, int startI, Func<int, int, double> assign, int startPosY)
            {
                for (var i = 0; i < inpCount; i++)
                {
                    var normalizedList = new double[neurons.Length];
                    var max = double.MinValue;
                    var min = double.MaxValue;
                    for (var j = 0; j < neurons.Length; j++)
                    {
                        normalizedList[j] = assign(i, j);
                        if (normalizedList[j] > max)
                            max = normalizedList[j];
                        if (normalizedList[j] < min)
                            min = normalizedList[j];
                    }

                    for (var j = 0; j < neurons.Length; j++)
                        normalizedList[j] = (normalizedList[j] - min) / (max - min);

                    for (var j = 0; j < neurons.Length; j++)
                    {
                        var pos = new Rectangle(j % s_recXCount * (s_recSize + s_recDif) + i % s_recLineCount * ((s_recSize + s_recDif) * s_recXCount + s_recBatchDif),
                                                j / s_recXCount * (s_recSize + s_recDif) + i / s_recLineCount * ((s_recSize + s_recDif) * (neurons.Length / s_recXCount)),
                                              s_recSize, s_recSize);
                        var points = FromCoordinates(pos.X + pos.Width / 2 + j / s_recXCount % 2 * s_recSize / 2, pos.Y - pos.Height / 4 * (j / s_recXCount) + pos.Height / 2 + startPosY, s_recSize);

                        var color = normalizedList[j] >= 0.5d ? Lerp(Color.Yellow, Color.Red, (normalizedList[j] - 0.5d) * 2d) : Lerp(Color.Blue, Color.Yellow, normalizedList[j] * 2d);
                        graphics.FillPolygon(new SolidBrush(color), points);
                    }

                    graphics.DrawString((i + startI).ToString(), font, Brushes.Black,
                        i % s_recLineCount * ((s_recSize + s_recDif) * s_recXCount + s_recBatchDif) + (s_recXCount - 1) * (s_recSize + s_recDif) / 2 - font.Size / 2f,
                        i / s_recLineCount * ((s_recSize + s_recDif) * neurons.Length / s_recXCount) + (neurons.Length - 1) / s_recXCount * (s_recSize + s_recDif) - font.Size / 2f + startPosY);
                }
            }

            Draw(inputsCount, 0, (i, j) => neurons[j][i], 0);
            Draw(inputs.Length, 2010, (i, j) => EuclidianDistance(neurons[j], inputs[i]), neurons.Length / s_recXCount * (s_recSize + s_recDif) + (inputsCount - 1) / s_recLineCount * ((s_recSize + s_recDif) * neurons.Length / s_recXCount));

            return image;
        }

        public static List<double> Convert(IEnumerable<string> value)
        {
            var ret = new List<double>();
            foreach (var str in value)
                ret.Add(double.Parse(str));

            return ret;
        }

        public static int[] GetCounts(int inputsCount)
        {
            var ret = new int[(int)(Math.Log(inputsCount) / Math.Log(2d)) - 1];

            ret[0] = inputsCount / 2;
            for (var i = 1; i < ret.Length; i++)
                ret[i] = ret[i - 1] / 2;

            return ret;
        }

        public static double[] Compute(string filePath)
        {
            var data = Convert(File.ReadLines(filePath));
            var network = new Network(new DistanceActivation(), data.Count, GetCounts(data.Count));

            for (var layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
            {
                var layer = network[layerIndex];
                for (var i = 0; i < layer.Neurons.Length; i++)
                {
                    var neuron = layer[i];
                    for (var j = 0; j < neuron.InputsCount; j++)
                        neuron[j] = j >= i * 2 && j <= i * 2 + 1 ? 1 : 0;
                }
            }

            return network.Compute(data.ToArray());
        }

        private static int s_recXCount;

        public static void Main()
        {
            var data = new string[]
            {
                "Data/2010.txt",
                "Data/2011.txt",
                "Data/2012.txt",
                "Data/2013.txt",
                "Data/2014.txt",
                "Data/2015.txt",
                "Data/2016.txt",
                "Data/2017.txt",
                "Data/2018.txt"
            };

            var settings = Settings1.Default;

            string input;
            do
            {
                Console.Clear();
                Console.WriteLine($"Neurons count: {settings.NeruonsCount} (1)\n" +
                    $"IterationsCount: {settings.IterationsCount} (2)\n" +
                    $"Sigma: {settings.Sigma} (3)\n" +
                    $"Alpha: {settings.Alpha} (4)\n\n" +
                    $"1 (n) (value) - Change parameter\n" +
                    $"2 - Train");

                input = Console.ReadLine();
                if (input.StartsWith("1"))
                {
                    var par = input.Split(' ');
                    switch (System.Convert.ToInt32(par[1]))
                    {
                        case 1:
                            settings.NeruonsCount = System.Convert.ToInt32(par[2]);
                            break;
                        case 2:
                            settings.IterationsCount = System.Convert.ToInt32(par[2]);
                            break;
                        case 3:
                            settings.Sigma = System.Convert.ToDouble(par[2]);
                            break;
                        case 4:
                            settings.Alpha = System.Convert.ToDouble(par[2]);
                            break;
                    }
                }
            }
            while (!input.StartsWith("2"));

            s_recXCount = settings.NeruonsCount;
            settings.Save();

            var epochData = new List<double[]>(data.Length);
            for (var i = 0; i < data.Length - 1; i++)
                epochData.Add(Convert(File.ReadAllLines(data[i])).ToArray());

            var network = new SelfOrganizingMap(epochData[0].Length, settings.NeruonsCount);
            var teacher = new SelfOrganizingMapTeacher(network);

            network.Randomize();
            var res = 10d;
            var maxres = double.MaxValue;
            Network minNetwork = null;
            var iterations = settings.IterationsCount;
            while (res > 1d && teacher.Iteration < iterations)
            {
                res = teacher.RunEpoch(epochData.ToArray(), null);
                if (res < maxres)
                {
                    maxres = res;
                    minNetwork = network.Copy();
                }
                Console.WriteLine(res);
            }

            epochData.Add(Convert(File.ReadLines(data[data.Length - 1])).ToArray());

            Console.WriteLine($"min value: {maxres}");
            using (var stream = new FileStream("image.png", FileMode.Create))
                Draw(minNetwork[0].Neurons, epochData.ToArray()).Save(stream, System.Drawing.Imaging.ImageFormat.Png);

            System.Diagnostics.Process.Start("image.png");
        }
    }
}