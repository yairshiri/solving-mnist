namespace ML
{
    public class Scalar:Tensor
    {
        #region variables

        public new float Data
        {
            get => _data[0];
            set => _data[0]= value;
        }

        #endregion
        
        
        #region  constructors
        // data, name constructor
        public Scalar(float data, string name) : base(0, name)
        {
            Data = data;
        }
        // data constructor
        public Scalar(float data) : base(0)
        {
            Data = data;
        }
        // empty constructor
        public Scalar() : base(0)
        {
            
        }
        // copy constructor 
        public Scalar(Scalar a) : base(0, a.Name)
        {
            Data = a.Data;
        }
        #endregion
        
        #region operators

        public static float operator +(Scalar a)
        {
            return a.Data;
        }
        public static float operator +(Scalar a,float b)
        {
            return a.Data + b;
        }
        public static float operator +(float a,Scalar b)
        {
            return b.Data + a;
        }  
        public static float operator -(Scalar a,float b)
        {
            return a.Data - b;
        }
        public static float operator -(float a,Scalar b)
        {
            return a-b.Data;
        }
        public static float operator *(Scalar a,float b)
        {
            return a.Data * b;
        }
        public static float operator *(Scalar a,Scalar b)
        {
            return a.Data * b.Data;
        }
        
        
        #endregion
    }
}