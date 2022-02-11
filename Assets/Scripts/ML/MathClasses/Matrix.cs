using System;
using UnityEngine.Assertions;

namespace ML
{
    public class Matrix:Tensor
    {
        #region variables

        

        public new Tensor[] Data
        {
            get => _data;
            set
            {
                _data = new Tensor[value.Length];
                for (int i = 0; i < _data.Length; i++)
                {
                    _data[i] = new Tensor(value[i]);
                }
                // setting the new shape
                Shape = new []{value.Length,value[0].Length};
            }
        }


        #endregion
        
        #region constructors
        // copy constructor
        public Matrix(Matrix a) : base(a)
        {
        }
        // init data to all value
        public Matrix(int height, int width,double value = 0,string name="") : base(new []{height,width},defaultValue:value,name:name)
        {
        }
        
        // copy from tensor 
        public Matrix(Tensor data) :base(data)
        {
        }
        
        #endregion

        #region operators
        // matrix multiplication. OoOOoOO! Scary!
        public static Matrix operator *(Matrix a, Matrix b)
        {
            // matrices need to be of size MxK,KxN
            Assert.AreEqual(a.Width,b.Height);
            // creating the ret value
            Matrix ret = new Matrix(a.Height,b.Width);
            // looping vertically on a
            for (int i = 0; i < a.Height; i++)
            {   
                // horizontally with b
                for (int j = 0; j < b.Width; j++)
                {   
                    // vertically with b and horizontally with a. limit of k can either be a.width or b.hieght
                    // since they are equal
                    for (int k = 0; k < a.Width; k++)
                    {
                        ret[i][j].Value += (a[i][k] * b[k][j]).Value;
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
                    ret[i][j].Value *= b;
                }
            }
            return ret;
        }
        // matrix element-wise multiplication
        public static Matrix operator *(float a, Matrix b)
        {
            return b*a;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            //making sure they have the same size
            Assert.AreEqual(a.Shape, b.Shape);
            // copying the first matrix
            Matrix ret = new Matrix(a);
            // adding the values of b to the copy of a
            for (int i = 0; i < ret.Height; i++)
            {
                for (int j = 0; j < ret.Width; j++)
                {
                    ret[i][j] += b[i][j];
                }
            }
            return ret;
        }
        public static Matrix operator +(Matrix a, Tensor b)
        {
            // make sure b is a matrix
            Assert.AreEqual(a.Dimension, b.Dimension);
            // copying the tensor
            Matrix ret = new Matrix(b);
            // adding the values of the matrix to the copy of the tensor
            for (int i = 0; i < ret.Height; i++)
            {
                for (int j = 0; j < ret.Width; j++)
                {
                    ret[i][j] += a[i][j];
                }
            }
            return ret;
        }
        public static Matrix operator +(Tensor a, Matrix b)
        {
            // make sure a is a matrix
            Assert.AreEqual(a.Dimension, b.Dimension);
            // copying the tensor
            Matrix ret = new Matrix(a);
            // adding the values of the matrix to the copy of the tensor
            for (int i = 0; i < ret.Height; i++)
            {
                for (int j = 0; j < ret.Width; j++)
                {
                    ret[i][j] += b[i][j];
                }
            }
            return ret;
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



        public override Tensor Clone()
        {
            return new Matrix(this);
        }

        public  Tensor Transpose()
        {
            // rotate the matrix by 90 degrees
            Matrix ret = new Matrix(this.Width, this.Height);
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    ret[j][i].Data = this[i][j].Data;
                }        
            }
            return ret;
        }

        public override bool Multiplyable(Tensor a)
        {
            // check dimensionality
            if (Math.Abs(a.Dimension - Dimension) > 1 && Dimension >= a.Dimension)
                return false;
            // check the size of the shape
            if (Width != a.Shape[0])
                return false;
            return true;
        }
        #endregion
    }
}