using System;

namespace ML
{
    public class SigmoidLayer:ActionLayer
    {
        #region Fields

        #endregion Fields
        #region Constructors
        public SigmoidLayer(int shape, string name = "") : base(new []{shape}, name)
        {
            var sigmoid = new Function<Tensor, Tensor>(SigmoidFunc, SigmoidDeriv, "Sigmoid");
            Init(sigmoid);
        }
        #endregion Constructors
        #region Methods
        private static Tensor SigmoidFunc(Tensor x)
        {
            //the sigmoid function
            return new Scalar(1/(1+(float)Math.Exp(-x.Data)));
        }

        private static Tensor SigmoidDeriv(Tensor x)
        {
            //the sigmoid derivative
            return new Scalar(SigmoidFunc(x).Data*(1-SigmoidFunc(x).Data));
        }

        #endregion Methods

    }
}