using System;

namespace ML
{
    public class LinearLayer:ActionLayer
    {
        #region Fields

        #endregion Fields
        #region Constructors
        public LinearLayer(int shape, string name = "") : base(new []{shape}, name)
        {
            var linear = new Function<Tensor, Tensor>(linearFunc, linearDeriv, "Linear");
            Init(linear);
        }
        #endregion Constructors
        #region Methods
        private static Tensor linearFunc(Tensor x)
        {
            return new Scalar(x.Data);
        }

        private static Tensor linearDeriv(Tensor x)
        {
            return new Scalar(1);
        }

        #endregion Methods

    }
}