double sigmoid(double x1, double x2, double w1, double w2, double b)
{
   double f = w1*x1 + w2*x2 + b; 
   return 1/(1+exp(-f)); 
} 
