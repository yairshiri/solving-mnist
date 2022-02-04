using UnityEngine.Assertions;

namespace ML
{
    public class Matrix:Tensor
    {
        #region variables

        private int _height;
        private int _width;
        private new Scalar[][] _data;


        public int Height
        {
            get => _height;
            private set => _height = value;
        }
        public int Width
        {
            get => _width;
            private set => _width = value;
        }

        public new Scalar[][] Data
        {
            get => _data;
            set
            {
                _data =(Scalar[][]) value.Clone();
                // setting the new length, width, height
                Length = value.Length;
                Width = _data[0].Length;
                Height = value.Length;
            }
        }


        #endregion
        
        #region constructors
        // name constructor, empty
        public Matrix(string name) : base(2, name)
        {
        }
        // empty constructor
        public Matrix() : base(2)
        {
        }
        
        // constructor with data and name
        public Matrix(Scalar[][] data, string name) : base(2, name)
        {
            // setting the data. we created the setter above
            Data = data;
        }
        
        // constructor with only data
        public Matrix(Scalar[][] data) : base(2)
        {
            // setting the data. we created the setter above
            Data = data;
        }
        // copy constructor
        public Matrix(Matrix a) : base(2, a.Name)
        {
            Data = a.Data;
        }
        // init data to all value
        public Matrix(int height, int width,float value) : base(2)
        {
            // initiating width and height
            Height = height;
            Width = width;
            // creating the data matrix
            _data = new Scalar[Height][];
            // setting the data to value
            for (int i = 0; i < Height; i++)
            {
                _data[i] = new Scalar[Width];
                for (int j = 0; j < Width; j++)
                {
                    _data[i][j] = new Scalar(value);
                }
            }
        }

        #endregion

        #region operators
        // matrix multiplication. OoOOoOO! Scary!
        public static Matrix operator *(Matrix a, Matrix b)
        {
            // matrices need to be of size MxK,KxN
            Assert.AreEqual(a.Height,b.Width);
            // creating the ret value
            Matrix ret = new Matrix(a.Width,b.Height,0);
            // looping vertically on a
            for (int i = 0; i < a.Width; i++)
            {   
                // horizontally with b
                for (int j = 0; j < b.Height; j++)
                {   
                    // vertically with b and horizontally with a. limit of k can either be a.width or b.hieght
                    // since they are equal
                    for (int k = 0; k < a.Height; k++)
                    {
                        ret[i][j].Data = a[i][k] * b[k][j];
                    }
                }
            }

            return ret;
        }
        // matrix element-wise multiplication
        public static Matrix operator *(Matrix a, float b)
        {
            Matrix ret = new Matrix(a);
            for (int i = 0; i < a.Height; i++)
            {
                for (int j = 0; j < a.Width; j++)
                {
                    ret[i][j] = new Scalar(ret[i][j] * b);
                }
            }
            return ret;
        }
        // matrix element-wise multiplication
        public static Matrix operator *(float a, Matrix b)
        {
            return b*a;
        }
        public new Scalar[] this[int i]
        {
            get => _data[i];
        }

        public  float this[int i, int j]
        {
            get => Data[i][j].Data;
            set => Data[i][j] = new Scalar(value);
        }
        
        #endregion

        #region methods

        public override string ToString()
        {
            // getting the base string from the base class (the name, if present)
            string ret = base.ToString();
            // adding the values
            for (int i = 0; i < Height; i++)
            {
                ret += "|";
                // adding the values
                for (int j = 0; j < Width; j++)
                {
                    ret += Data[i][j].Data + "|";
                }
                // adding another new line
                ret += "\n";
            }
            


            return ret;
            
        }
        

        #endregion
    }
}