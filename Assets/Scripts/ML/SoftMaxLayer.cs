using System;

namespace ML
{
    public class SoftMaxLayer:ActionLayer
    {
        #region Fields

        private static float NOISE = 0.00001f;

        #endregion
        
        public SoftMaxLayer(int size, string name = "") : base(new [] {size},name,false)
        {
            var softmax = new Function<Tensor, Tensor>(softmaxFunc, softmaxDeriv,"softmax");
            Init(softmax);
        }

        #region Methods

        private static Tensor softmaxFunc(Tensor x) 
        {
            // initializing a new vector with the length of x.
            Vector ret = new Vector(x.Length);
            float sum = 0;
            // finding the max value in x
            float max = x[0];
            for (int i = 1; i < ret.Length; i++)
                if (x[i] > max)
                    max = x[i];
            // every element is e to the power of the elements devided by the sum of e to the power of all the elements.
            // implementation from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            for (int i = 0; i < ret.Length; i++)
            {   
                // x = e^x
                ret[i] = (float)Math.Exp(x[i]-max);
                // adding to the sum
                sum += ret[i];
            }
            // deviding by the sum
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] /= sum;
                //we do this because we dont want to have 0s (for backprop), so we set a lower bound (NOISE)
                ret[i] = Math.Max(ret[i], NOISE);
            }
            return ret;
        }
        // softmax deriv from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        private Tensor softmaxDeriv(Tensor x)
        {
            //copying x and getting the softmax values for the deriv
            Vector ret = (Vector)softmaxFunc((Vector)_neuronActivations);
            for (int i = 0; i < x.Length; i++)
            {
                // ret[i] = the derivative of Si w.r.t Zi
                ret[i] *= 1-ret[i];
            }
            return ret;
        }



        #endregion
        
    }
}