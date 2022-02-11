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
    private static int sampleSize = 100000;
    private static int batchSize = 100;

    private float noise = 0.00001f;
    
    
    // counts loops through update
    private int counter = 0;
    // log every LOG_INTERVAL loops through update
    private int LOG_INTERVAL = 20;
    
    private float lr = 0.00001f;
    private Tensor[] features = new Tensor[sampleSize];
    private Tensor[] labels = new Tensor[sampleSize];
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
    private ML.Loss BCE = new Loss( BCCEFunc,CEDeriv,"binary categorical crossentropy");
    private Network net;
    
    
    // Start is called before the first frame update
    void Start()
    {
        x = (float)rand.NextDouble() * 10;
        // generating the data
        Debug.Log("Generating data...");
        for (int i = 0; i < sampleSize; i++)
        {
            features[i] = new Vector(1, x,"Data "+i);
            labels[i] = new Vector(1, x * x * x + 6 * x + 2, "Label " + i);
            x = (float)rand.NextDouble() * 10;

        }

        Debug.Log("Done!");
        Layer[] layers=
        {
            new DenseLayer(3,"d1"),
            new ReLULayer(3,"activation 1"),
            new DenseLayer(4,"d2"),
            new ReLULayer(4,"activation 2"),
            new DenseLayer(labels[0].Length,"d3"),
            new LinearLayer(labels[0].Length,"activation 3")
        };
        net = new Network(layers,lr,1,mse);

    }
    #region activations
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
    
    private static Tensor mseFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Vector ret = new Vector(1);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[0] += (float)Math.Pow(input.pred[i] - input.label[i],2);
        }

        return ret;
    }

    private static Vector mseDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Vector ret = new Vector(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i] = 2*(input.pred[i] - input.label[i]);
        }
        return ret;

    }

    private static Vector CEFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Vector ret = new Vector(1);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[0] += -input.label[i] * (float)Math.Log(input.pred[i]);
        }
        return ret;
    }

    private static Tensor CEDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        
        Vector ret = new Vector(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i] = -(input.label[i] / input.pred[i]);
        }
        return ret;

    }
    private static Vector BCCEFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Vector ret = new Vector(1);
        //L = -t1*log(s1) - (1-t1)*log(1-s1)
        ret[0] = -input.label[0] * (float)Math.Log(input.pred[0]) - input.label[1] * (float)Math.Log(input.pred[1]);
        return ret;
    }

    private static Tensor BCCEDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Vector ret = new Vector(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i] = -(input.label[i] / input.pred[i]) + (1 - input.label[i])/(1-input.pred[i]);
        }
        return ret;

    }
    
    #endregion
    
    // a method that does the logging. yeah it's a great comment.
    void log()
    {
        Debug.Log(net.ToString());
    }
    
    
    // Update is called once per frame
    void Update()
    {
        x = (float)rand.NextDouble() * 10;
        net.backwards(features,labels,batchSize);
        counter++;
        if (counter % LOG_INTERVAL == 0)
        {
            log();
        }
        
    }
}
