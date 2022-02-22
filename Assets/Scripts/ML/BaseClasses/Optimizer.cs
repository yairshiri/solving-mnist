using System;

namespace ML
{
    public abstract class Optimizer
    {
        protected static Random rand = new Random();
        protected private double learningRate = 0.00001;
        public abstract (Tensor[],Tensor[]) backwards(Tensor[] features, Tensor[] labels);

        public delegate (Tensor[], Tensor[]) getGrad(Tensor a, Tensor b);

        public static getGrad GetGrad;

    }
}