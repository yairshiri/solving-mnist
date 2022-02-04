using System;

namespace ML
{
    public class ReLULayer:ActionLayer
    {
        #region Fields

        #endregion Fields
        #region Constructors
        public ReLULayer(int shape, string name = "") : base(new []{shape}, name)
        {
            var reLu = new Function<Tensor, Tensor>(reluFunc, reluDeriv, "ReLU");
            Init(reLu);
        }
        #endregion Constructors
        #region Methods
        private static Tensor reluFunc(Tensor x)
        {
            return new Scalar(Math.Max(x.Data, 0));
        }

        private static Tensor reluDeriv(Tensor x)
        {
            float ret = 0;
            // the relu derivative: 
            //{ x > 0 : 1}
            //{ x <= 0: 0}
            if (x.Data > 0)
                ret = 1;
            return new Scalar(ret);
        }

        #endregion Methods

    }
}