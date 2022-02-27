using UnityEngine;

namespace ML
{
    public class Flatten:ActionLayer
    {

        private int axis;
        public Flatten(int axis,string name = "") : base(new int[]{}, name, false)
        {
            this.axis = axis;
            base.Init(new Function<Tensor, Tensor>(FlattenFunc,FlattenDeriv,"Flatten layer"));
        }
        
        public override void Init(int[] shape)
        {
            base.Init(shape);
            outputShape = new[]{1};
            for (int i = 0; i < inputShape.Length; i++)
            {
                outputShape[0] *= inputShape[i];
            }
        }


        public  Tensor FlattenFunc(Tensor x)
        {
            Tensor ret = new Tensor(x.NumOfElements);
            if (axis == 1)
                x = x.Transpose();
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j <x[0].Shape[0]; j++)
                {
                    for (int k = 0; k < x[0][0].Shape[0]; k++)
                    { 
                        ret[i*x[0].NumOfElements + j*x[0][0].NumOfElements + k].Value = x[i][j][k].Value;
                    }
                }
            }
            return ret;
        }


        public Tensor FlattenDeriv(Tensor x)
        {
            return new Tensor(value:1.0);
        }

        public override Tensor bPass(Tensor loss)
        {
            return NeuronActivations;
        }
    }
}