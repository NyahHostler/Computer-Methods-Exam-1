import math
from random import random as rnd
from copy import deepcopy as dc
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

#region functions
def ln_PDF(args):
    """
    Computes the log-normal probability density function.
    :param args: tuple (D, mu, sig)
    :return: f(D)
    """
    import numpy as np
    D, mu, sig = args
    D = np.asarray(D).item()
    if D == 0.0:
        return 0.0
    p = 1/(D*sig*math.sqrt(2*math.pi))
    _exp = -((math.log(D)-mu)**2)/(2*sig**2)
    return p*math.exp(_exp)

def tln_PDF(args):
    """
    Computes the truncated log-normal probability density function.
    :param args: tuple (D, mu, sig, F_DMin, F_DMax)
    :return: f_trunc(D)
    """
    import numpy as np
    D, mu, sig, F_DMin, F_DMax = args
    D = np.asarray(D).item()
    return ln_PDF((D, mu, sig))/(F_DMax-F_DMin)

def F_tlnpdf(args):
    """
    Computes the CDF of the truncated log-normal distribution by integrating
    the truncated PDF from D_Min to D.
    :param args: tuple (mu, sig, D_Min, D_Max, D, F_DMax, F_DMin)
    :return: cumulative probability P
    """
    import numpy as np
    mu, sig, D_Min, D_Max, D, F_DMax, F_DMin = args
    D = np.asarray(D).item()
    if D > D_Max or D < D_Min:
        return 0
    '''
    replaced with quad function
    '''
    P, _ = quad(lambda x: tln_PDF((x, mu, sig, F_DMin, F_DMax)), D_Min, D)
    return P

def makeSample(args, N=100):
    """
    Generates a random sample of N values from a truncated log-normal distribution
    using inverse transform sampling.
    :param args: tuple (ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin)
    :param N: number of values in the sample
    :return: list of N sampled values
    """
    ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin = args
    probs = [rnd() for _ in range(N)]
    '''
    Replaced with brentq for reliable bracketed root finding between D_Min and D_Max.
    Probabilities clamped slightly away from 0 and 1 to ensure valid bracketing.
    '''
    epsilon = 1e-9
    a = D_Min + epsilon
    b = D_Max - epsilon
    d_s = []
    for i in range(len(probs)):
        p = max(epsilon, min(1 - epsilon, probs[i]))
        fa = F_tlnpdf((ln_Mean, ln_sig, D_Min, D_Max, a, F_DMax, F_DMin)) - p
        fb = F_tlnpdf((ln_Mean, ln_sig, D_Min, D_Max, b, F_DMax, F_DMin)) - p
        if fa * fb < 0:
            d_s.append(brentq(
                lambda D: F_tlnpdf((ln_Mean, ln_sig, D_Min, D_Max, D, F_DMax, F_DMin)) - p,
                a, b
            ))
        else:
            d_s.append((D_Min + D_Max) / 2)  # fallback to midpoint if bracket fails
    return d_s

def sampleStats(D, doPrint=False):
    """
    Computes the mean and variance of a list of values.
    :param D: list of values
    :param doPrint: bool, prints results if True
    :return: (mean, var)
    """
    N = len(D)
    mean = sum(D)/N
    var = 0
    for d in D:
        var += (d-mean)**2
    var /= N-1
    if doPrint == True:
        print(f"mean = {mean:0.3f}, var = {var:0.3f}")
    return (mean, var)

def getDistributionParameters(args):
    """
    Prompts the user to input the mean and standard deviation of the log-normal distribution.
    :param args: default values (mean_ln, sig_ln)
    :return: (mean_ln, sig_ln)
    """
    mean_ln, sig_ln = args
    st_mean_ln = input(f'Mean of ln(D)? (ln({math.exp(mean_ln):0.1f})={mean_ln:0.3f}):').strip()
    mean_ln = mean_ln if st_mean_ln == '' else float(st_mean_ln)
    st_sig_ln = input(f'Standard deviation of ln(D)? ({sig_ln:0.3f}):').strip()
    sig_ln = sig_ln if st_sig_ln == '' else float(st_sig_ln)
    return (mean_ln, sig_ln)

def getTruncationParameters(args):
    """
    Prompts the user to input the upper and lower truncation bounds.
    :param args: (D_Min, D_Max)
    :return: (D_Min, D_Max)
    """
    D_Min, D_Max = args
    st_D_Max = input(f'Upper truncation bound D_Max? ({D_Max:0.3f})').strip()
    D_Max = D_Max if st_D_Max == '' else float(st_D_Max)
    st_D_Min = input(f'Lower truncation bound D_Min? ({D_Min:0.3f})').strip()
    D_Min = D_Min if st_D_Min == '' else float(st_D_Min)
    return (D_Min, D_Max)

def getSampleParameters(args):
    """
    Prompts the user to input the number of samples and sample size.
    :param args: (N_samples, N_sampleSize)
    :return: (N_samples, N_sampleSize)
    """
    N_samples, N_sampleSize = args
    st_N_Samples = input(f'How many samples? ({N_samples})').strip()
    N_samples = N_samples if st_N_Samples == '' else int(st_N_Samples)
    st_N_SampleSize = input(f'How many items in each sample? ({N_sampleSize})').strip()
    N_sampleSize = N_sampleSize if st_N_SampleSize == '' else int(st_N_SampleSize)
    return (N_samples, N_sampleSize)

def getFDMaxFDMin(args):
    """
    Computes the cumulative log-normal probabilities at D_Min and D_Max.
    :param args: tuple (mean_ln, sig_ln, D_Min, D_Max)
    :return: (F_DMin, F_DMax)
    """
    mean_ln, sig_ln, D_Min, D_Max = args
    '''
    replaced with quad
    '''
    F_DMax, _ = quad(lambda D: ln_PDF((D, mean_ln, sig_ln)), 0, D_Max)
    F_DMin, _ = quad(lambda D: ln_PDF((D, mean_ln, sig_ln)), 0, D_Min)
    return (F_DMin, F_DMax)

def makeSamples(args):
    """
    Generates multiple samples from the truncated log-normal distribution
    and computes the mean of each sample.
    :param args: (mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, doPrint)
    :return: Samples, Means
    """
    mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, doPrint = args
    Samples = []
    Means = []
    for n in range(N_samples):
        sample = makeSample((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin), N=N_sampleSize)
        Samples.append(sample)
        sample_Stats = sampleStats(sample)
        Means.append(sample_Stats[0])
        if doPrint == True:
            print(f"Sample {n}: mean = {sample_Stats[0]:0.3f}, var = {sample_Stats[1]:0.3f}")
    return Samples, Means

def main():
    '''
    This program simulates sampling from a truncated log-normal distribution.
    It produces N_samples samples of size N_sampleSize and reports the mean
    and variance of each sample, as well as the mean and variance of the sampling mean.
    Step 1: use input to get mean of ln(D), stdev of ln(D), D_Max, D_Min, N_samples, N_sampleSize
    Step 2: use random to produce uniformly distributed probability values and the truncated log-normal PDF to get values for D
    Step 3: compute the mean and variance of each sample and report to user
    Step 4: compute the mean and variance of the sampling mean and report to user
    :return: nothing
    '''
    # setup some default values
    mean_ln = math.log(2)
    sig_ln = 1
    D_Max = 1
    D_Min = 3.0/8.0
    N_samples = 11
    N_sampleSize = 100
    goAgain = True

    while (goAgain == True):
        # Step 1: get distribution and truncation parameters from user
        mean_ln, sig_ln = getDistributionParameters((mean_ln, sig_ln))
        D_Min, D_Max = getTruncationParameters((D_Min, D_Max))
        N_samples, N_sampleSize = getSampleParameters((N_samples, N_sampleSize))
        F_DMin, F_DMax = getFDMaxFDMin((mean_ln, sig_ln, D_Min, D_Max))

        #region plotting to check results
        # x = [_x*0.1 for _x in range(0,100)]
        # y = [ln_PDF((_x,ln_Mean, ln_sig)) for _x in x]
        # x_trunc = [D_Min+_x*(D_Max-D_Min)/99 for _x in range(100)]
        # y_trunc = [ln_PDF((_x,ln_Mean, ln_sig))/(F_DMax-F_DMin) for _x in x_trunc]
        #
        # fig, ax1= plt.subplots()
        # ax1.plot(x,y)
        # ax2=ax1.twinx()
        # ax2.plot(x_trunc,y_trunc)
        # plt.show()
        #endregion

        # Step 2 & 3: generate samples and report means and variances
        Samples, Means = makeSamples((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, True))

        # Step 4: compute and report mean and variance of the sampling mean
        stats_of_Means = sampleStats(Means)
        print(f"Mean of the sampling mean:  {stats_of_Means[0]:0.3f}")
        print(f"Variance of the sampling mean:  {stats_of_Means[1]:0.6f}")
        goAgain = input('Go again? (No)').strip().lower().__contains__('y')

#endregion

if __name__ == '__main__':
    main()