namespace ML
{
    public class Flatten:ActionLayer
    {

        private int axis;
        private Tensor deltaVals;
        public Flatten(int axis,string name = "") : base(new int[]{}, name, false)
        {
            this.axis = axis;
            base.Init(new Function<Tensor, Tensor>(FlattenFunc,FlattenDeriv,"Flatten layer"));
        }
        
        public override void Init(int[] shape)
        {
            base.Init(inputShape);
            outputShape = new[]{1};
            for (int i = 0; i < inputShape.Length; i++)
            {
                outputShape[0] *= inputShape[i];
            }
        }


        public  Tensor FlattenFunc(Tensor x)
        {
            Tensor ret = new Tensor(x.NumOfElements);
            deltaVals = x;
            if (axis == 1)
                x = x.Transpose();
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j <x[0].Shape[0]; j++)
                {
                    for (int k = 0; k < x[0][0].Shape[0]; k++)
                    {
                        ret[i * x.Shape[0] + j * x[0].Shape[0] + k] = x[i][j][k];
                    }
                }
            }
            return ret;
        }


        public Tensor FlattenDeriv(Tensor x)
        {
            return deltaVals;
        }
        
    }
}