using System;
using System.Collections;
using System.Collections.Generic;
using ML;
using UnityEngine;
using Network = ML.Network;
using Vector = ML.Vector;
using Random = System.Random;

public class MLMain : MonoBehaviour
{
    private int sampleSize = 100000;

    private float noise = 0.00001f;
    
    
    // counts loops through update
    private int counter = 0;
    // log every LOG_INTERVAL loops through update
    private int LOG_INTERVAL = 20;
    
    private float lr = 0.0001f;
    private Vector features = new Vector(1);
    private Vector labels = new Vector(1);
    private Random rand = new Random();
    private float x ;

    // creating the relu activation func 
    private ML.Activation relu = new Activation( reluFunc,reluDeriv,"relu");
    // creating the relu activation func 
    private ML.Activation sigmoid = new Activation( sigFunc,sigDeriv,"sigmoid");
    // creating the linear activation func 
    private ML.Activation linear = new Activation( linearFunc,linearDeriv,"linear");
    // creating the softmax activation func 
    //private ML.Function<Vector,Vector> softmax = new Function<Vector,Vector>( softmaxFunc,softmaxDeriv,"softmax");
    // creating the mse loss
    private ML.Loss mse = new Loss( mseFunc,mseDeriv,"mse");
    // creating the Categorical crossEntropy loss
    private ML.Loss CE = new Loss( CEFunc,CEDeriv,"categorical crossentropy");
    // creating the Binary Catergorical crossEntropy loss
    private ML.Loss BCE = new Loss( BCCEFunc,BCCEDeriv,"binary categorical crossentropy");
    private Network net;
    
    
    // Start is called before the first frame update
    void Start()
    {
        Layer[] layers=
        {
            
            new DenseLayer(3,relu,"d1"),
            new DenseLayer(4,relu,"d2"),
            new DenseLayer(1,linear,"output")};
        net = new Network(layers,lr,1,mse);
        x = (float)rand.NextDouble() * 10;
    }

    private static float reluFunc(float x)
    {
        return Math.Max(x, 0);
    }

    private static float sigFunc(float x)
    {
        return  1.0f / (float)(1 + Math.Exp(-x));
    }
    private static float sigDeriv(float x)
    {
        return sigFunc(x) * (1-sigFunc(x));
    }

    private static float reluDeriv(float x)
    {
        return Math.Max(x, 0)/Math.Abs(x);
    }
    private static float linearFunc(float x)
    {
        return x;
    }

    private static float linearDeriv(float x)
    {
        return 1;
    }
    
    private static Tensor mseFunc((Tensor x, Tensor y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        Vector ret = new Vector(1);
        for (int i = 0; i < input.x.Length; i++)
        {
            ret[0] += (float)Math.Pow(input.x[i] - input.y[i],2);
        }

        return ret;
    }

    private static Vector mseDeriv((Tensor x, Tensor y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        Vector ret = new Vector(input.x.Length);
        for (int i = 0; i < input.x.Length; i++)
        {
            ret[i] = 2*(input.x[i] - input.y[i]);
        }
        return ret;

    }

    private static Vector CEFunc((Tensor x, Tensor y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        Vector ret = new Vector(1);
        for (int i = 0; i < input.x.Length; i++)
        {
            ret[0] += -input.y[i] * (float)Math.Log(input.x[i]);
        }
        return ret;
    }

    private static Tensor CEDeriv((Tensor x, Tensor y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        
        Vector ret = new Vector(input.x.Length);
        for (int i = 0; i < input.x.Length; i++)
        {
            ret[i] = -(input.y[i] / input.x[i]) + (1 - input.y[i])/(1-input.x[i]);
        }
        return ret;

    }
    private static Vector BCCEFunc((Tensor x, Tensor y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        Vector ret = new Vector(1);
        //L = -t1*log(s1) - (1-t1)*log(1-s1)
        ret[0] = -input.y[0] * (float)Math.Log(input.x[0]) - (1 - input.y[0]) * (float)Math.Log(1 - input.x[0]);
        return ret;
    }

    private static Tensor BCCEDeriv((Tensor x, Tensor y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        Vector ret = new Vector(input.x.Length);
        for (int i = 0; i < input.x.Length; i++)
        {
            ret[i] = -(input.y[i] / input.x[i]) + ((1 - input.y[i])/(1-input.x[i]));
        }
        return ret;

    }
    
    // a method that does the logging
    void log()
    {
        Debug.Log(net.ToString());
    }
    
    
    // Update is called once per frame
    void Update()
    {
        x = (float)rand.NextDouble() * 10;
        features[0] = x;
        /*if (x > 5)
        {
            labels[0] = 1;
            labels[1] = 0;
        }
        else
        {
            labels[0] = 0;
            labels[1] = 1;

        }*/
        labels[0] = x*x + 6*x + 2;
        net.backwards(features,labels);
        counter++;
        if (counter % LOG_INTERVAL == 0)
        {
            log();
        }
        
    }
}
