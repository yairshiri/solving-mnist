﻿using System;
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
                _data = new Scalar[value.Length][];
                for (int i = 0; i < _data.Length; i++)
                {
                    _data[i] = new Scalar[value[i].Length];
                    for (int j = 0; j < value[0].Length; j++)
                    {
                        _data[i][j] = new Scalar(value[i][j]);
                    }
                }
                // setting the new length, width, height
                Length = value.Length;
                Width = _data[0].Length;
                Height = value.Length;
            }
        }


        #endregion
        
        #region constructors
        // name constructor, empty
        public Matrix(string name="") : base(2, name)
        {
            
        }
        // constructor with data and name
        public Matrix(Scalar[][] data, string name = "") : base(2, name)
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
        public Matrix(int height, int width,float value = 0) : base(2)
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
        
        // copy from tensor 
        public Matrix(Tensor data,bool copySize = false) :base(2,data.Name)
        {
            // make sure data is a Matrix
            Assert.AreEqual(data.Dimension, 2);
            // if we copy the data aswell as the sizes:
            if(copySize == false)
                Data = ((Matrix)data).Data;
            else
            {
                // if we only copy the sizes
                Height = ((Matrix)data).Height;
                Width = ((Matrix)data).Width;
                Length = ((Matrix)data).Length;
                // init the data
                // creating the data matrix
                _data = new Scalar[Height][];
                // setting the data to value
                for (int i = 0; i < Height; i++)
                {
                    _data[i] = new Scalar[Width];
                    for (int j = 0; j < Width; j++)
                    {
                        _data[i][j] = new Scalar(0);
                    }
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
            Matrix ret = new Matrix(a.Width,b.Height);
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

        public static Matrix operator +(Matrix a, Matrix b)
        {
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

        public override Tensor ElementWiseFunction(Func<Tensor, Tensor> func)
        {
            // creating a copy of this
            Matrix ret = new Matrix(this);
            // applying the function
            for (int i = 0; i < ret.Height; i++)
            {
                for (int j = 0; j < ret.Width; j++)
                {
                    ret[i][j] = new Scalar(func(ret.Data[i][j]));

                }
            }
            return ret;

        }

        public override Tensor ElementWiseMultiply(Tensor a)
        {
            // make sure they are both the same size
            Assert.AreEqual(a.Dimension,2);
            // convert a to a matrix
            Matrix currectA = new Matrix(a);
            // make sure they are of the same size
            Assert.AreEqual(currectA.Width,this.Width );
            Assert.AreEqual(currectA.Height,this.Height );
            // create the return Matrix
            Matrix ret = new Matrix(Height, Width);
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    // the multiplication
                    ret[i][j] = new Scalar(currectA[i][j] * this[i][j]);
            }
            return ret;
        }

        public override Tensor Clone()
        {
            return new Matrix(this);
        }

        public override Tensor Transpose()
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


        #endregion
    }
}