import numpy as np
import scipy.signal as sig
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings

class Signal():
    def __init__(self, samples=None, start=0, period=1):
        """
        Create a signal from its samples.

        Patameters:

            samples : ndarray
                sampled values of signal

            start : float or int
                start time of signal

            period : float or int
                time between samples
        """
        self._samples = samples
        self._start  = start
        self._period = period

    @property
    def samples(self):
        """
        Return the sample values of the signal.
        """
        return self._samples

    @property
    def n_samples(self):
        """
        Number of samples of the signal.
        """
        return self.samples.shape[0]
    
    @property
    def period(self):
        """
        Return the period between samples of the signal, in seconds.
        """
        return self._period
    
    @period.setter
    def period(self,value):
        return self._period
        
    @property
    def size(self):
        """
        Number of samples in the signal.
        """
        return self._samples.size

    @property
    def shape(self):
        """Shape of sample array. By convention, the first dimensions should always be
the time index.
        """
        return self.samples.shape
    
    @property
    def rate(self):
        """Sample rate of the signal in Hz"""
        return 1/self.period
        
    @property
    def end(self):
        """
        End of the signal, in seconds.
        """
        return self.start + (self.size-1)*self.period
    
    @property
    def start(self):
        """
        Start time of the signal, in seconds
        """
        return self._start

    @start.setter
    def start(self, value):
        self._start = value
    
    @property
    def duration(self):
        return self.size * self.period
    
    @duration.setter
    def duration(self,value):
        """
        Set the total duration of the signal, in seconds.  

        Note: changes the signal's period, rate, and end.
        """
        self._period = value / self.size
    
    @property
    def time(self):
        """
        Array of sample times.
        """
        return self.start + np.arange(self.shape[0])*self.period
    
    def plot(self, ax=None, **kwds):
        if not ax:
            ax = plt.gca()

        if len(self.shape) == 1:
            ax.plot(self.time,self.samples)
        elif len(self.shape) == 2:
            n_channels = self.shape[1]
            for chan in range(n_channels):
                channel_samples = self.samples[:,chan]
                ax.plot(self.time,channel_samples)
        else:
            pass

        return ax
            
    def apply(self,f,scale=1,loc=0):
        """
        Given a signal $x(t)$, a function $f:R->R$, compute $y = f(a*x)$

        y = f((x-loc)/scale)
        
        Parameters:
            f : function
            scale : float or int
        Returns:
            y : signal
        """
        new_samples = f((self.samples - loc)/scale)
        return Signal(new_samples,self.start,self.period)
        
    def convolve(self,h,**kwds):
        if not np.isclose(self.period,h.period):
            warnings.warn("signal periods differ")
            
        period = self.period
        start = self.start + h.start
        samples = sig.convolve(self.samples,h.samples,**kwds) * period
        return Signal(samples,start,period)
    
    def matched_filter(self,symbol_period):
        start = -self.end + symbol_period
        samples = np.flip(self.samples,0)
        return Signal(samples,start,self.period)
    
    def modulate(self,symbols,symbol_rate):
        """
        Modulate a sequence of amplitudes at the given symbol rate.

        Parameters:
            symbols : ndarray, float or int
                sequence of symbols, i.e., amplitudes, to modulate.
            symbol_period : foat or int
                period between symbols.
        Returns:
            y : Signal
                Modulated signal
        """
        return 0
    
def trange(n_samples, start = 0, period = 1):
    """"
    Return the identity signal, i.e., x(t) = t
    
    Parameters:
        start : float
            start time of the signal, in seconds
        period : float
            period between samples of the signal, in seconds
    """
    x = start + np.arange(n_samples) * period
    return Signal(x,start,period)

def lowpass_filter(N,W,Ts):
    start = -N/2*Ts
    t = (np.arange(N)-N/2)*Ts
    x = 2*W*np.sinc(2*W*t)
    return Signal(x,start,Ts)

def rcos(t,alpha=0.35):
    if np.isclose(alpha,0):
        y = np.sinc(t)
    else:
        f1 = lambda x: np.pi/4*np.sinc(1/(2*alpha))
        f2 = lambda x: np.sinc(x) * np.cos(np.pi*alpha*x) / (1 - (2*alpha*x)**2)
        cond1 = np.isclose(np.abs(t),1/(2*alpha))
        cond2 = ~cond1
        y = np.piecewise(t,[cond1,cond2],[f1,f2])
    return y

def rrcos(t,alpha=0.35):
    """
    Raised cosine pulse. Note: \[ Hello \] $Hello$
    """
    if np.isclose(alpha,0):
        y = np.sinc(t)
    else:
        b0 = 1+2/np.pi
        b1 = 1-2/np.pi
        c0 = np.pi*(1-alpha)
        c1 = np.pi*(1+alpha)
        s0 = 4*alpha
        cond1 = np.isclose(t,0)
        cond2 = np.isclose(np.abs(t),1/s0)
        cond3 = ~(cond1|cond2)
        f1 = lambda x: 1+alpha*(4/np.pi - 1)
        f2 = lambda x: alpha/np.sqrt(2) * (b0*np.sin(np.pi/s0)+b1*(np.cos(np.pi/s0)))
        f3 = lambda x: (np.sin(c0*x) + s0*x*np.cos(c1*x)) / (np.pi*x*(1-(s0*x)**2))
        y = np.piecewise(t,[cond1,cond2,cond3],[f1,f2,f3])
    return y

def comb(t,T,**kwds):
    r = np.mod(t,T)
    y = (np.isclose(r,0,**kwds)|np.isclose(r,T,**kwds))
    return y.astype(t.dtype)

def id(n_samples,period=1.0,start=0):
    samples = start + np.arange(n_samples) * period
    return Signal(samples,start,period)

def qfunc(x): return stats.norm.sf(x,loc=0,scale=1)
def qfuncinv(x): return stats.norm.ppf(-x + 1)

