import math
from scipy.stats import norm  # type: ignore
import numpy as np
import warnings

NORMAL = norm(0, 1)
SQRT3 = 1.732050807568877293527446341505872
VOLTOL = 1e-8


def b(x: float, vol: float, is_call: bool) -> float:
    assert vol > 1e-16
    sgn = 1.0 if is_call else -1.0
    if np.abs(vol) < VOLTOL:
        warnings.warn("vol is below tolerance")
        if x > 0:
            n1, n2 = 1.0, 1.0
        else:
            n1, n2 = 0.0, 0.0
    else:
        arg1 = sgn * (x / vol + vol / 2)
        arg2 = sgn * (x / vol - vol / 2)
        n1 = NORMAL.cdf(arg1)  # type: ignore
        n2 = NORMAL.cdf(arg2)  # type: ignore
    out = sgn * (math.exp(x / 2) * n1 - math.exp(-x / 2) * n2)  # type: ignore
    out = float(out)  # type: ignore
    if np.abs(out) < 1e-16:
        warnings.warn("b quote is zero")
    return out


class OptionData:
    def __init__(self, x: float, is_call: bool):
        self.x = x
        self.is_call = is_call
        self.sigmaL = 0.0
        self.sigmaC = 0.0
        self.sigmaU = 0.0
        self.bL = 0.0
        self.bC = 0.0
        self.bU = 0.0
        self.bLP = 0.0
        self.bCP = 0.0
        self.bUP = 0.0
        self.bMax = 0.0
        self.quoteShift = 0.0  # TODO: use this to requote OTM call

        self.init_values()
        self.init_approx_center_left()
        self.init_approx_center_right()
        self.init_approx_lower()
        self.init_approx_upper()

    def bprime(self, vol: float) -> float:
        a = self.x / vol
        b = vol / 2
        out = 1 / (math.sqrt(2) * math.sqrt(math.pi)) * math.exp(-0.5 * (a * a + b * b))
        if np.abs(out) < 1e-16:
            warnings.warn(f"bprime zero for {vol} and x={self.x}")
        return out

    def init_values(self):
        s = 2 * int(self.is_call) - 1

        # should use "parity" identities in paper to reduce to OTM call
        # TODO here
        # for the moment, just panic if not possible

        # at this point, we need to be in a OTM call situation
        assert self.is_call
        assert self.x <= 0.00

        self.bMax = math.exp(s * self.x / 2)
        self.sigmaC = math.sqrt(2 * abs(self.x))
        self.bC = b(self.x, self.sigmaC, self.is_call)
        self.bCP = self.bprime(self.sigmaC)
        self.sigmaU = self.sigmaC + (self.bMax - self.bC) / self.bCP
        self.bU = b(self.x, self.sigmaU, self.is_call)
        self.bUP = self.bprime(self.sigmaU)
        self.sigmaL = self.sigmaC - self.bC / self.bCP
        self.bL = b(self.x, self.sigmaL, self.is_call)
        self.bLP = self.bprime(self.sigmaL)

    def init_approx_center_left(self) -> None:
        self.izacl = rational_auto_right(
            self.bL, self.bC, self.sigmaL, self.sigmaC, 1 / self.bLP, 1 / self.bCP, 0.0
        )

    def init_approx_center_right(self) -> None:
        self.izacr = rational_auto_left(
            self.bC, self.bU, self.sigmaC, self.sigmaU, 1 / self.bCP, 1 / self.bUP, 0.0
        )

    def init_approx_upper(self) -> None:
        x = self.x
        fu = float(NORMAL.cdf(-self.sigmaU / 2.0))  # type: ignore
        x2 = x * x
        s2 = self.sigmaU * self.sigmaU
        su = -0.5 * math.exp(0.5 * x2 / s2)
        ssu = (
            1
            / self.sigmaU
            * math.sqrt(math.pi)
            / math.sqrt(2)
            * x2
            / s2
            * math.exp(x2 / s2 + s2 / 8.0)
        )
        r_auto = r_auto_left(self.bMax - self.bU, fu, 0.0, su, -0.5, ssu)
        g = rational(self.bU, self.bMax, fu, 0.0, su, -0.5, r_auto)

        def f(beta: float) -> float:
            return -2 * float(NORMAL.ppf(g(beta)))  # type: ignore

        self.izau = f

    def init_approx_lower(self) -> None:
        f, fp, fpp = lower_func(self.x, self.sigmaL)
        fun = rational_auto_right(0, self.bL, 0.00, f, 1.00, fp, fpp)

        def out(quote: float) -> float:

            denom = float(
                NORMAL.ppf(  # type: ignore
                    SQRT3 * np.cbrt(fun(quote) / (2 * np.pi * np.abs(self.x)))
                )  # type: ignore
            )
            return float(np.abs(self.x / SQRT3 * 1 / denom))

        self.izal = out

    def iter_zero_approx(self, quote: float) -> float:
        if quote <= 0.00:
            raise ValueError("price must be positive")

        if quote <= self.bL:
            return self.izal(quote)

        if quote <= self.bC:
            return self.izacl(quote)

        if quote <= self.bU:
            return self.izacr(quote)

        if quote <= self.bMax:
            return self.izau(quote)

        raise ValueError("quote exceeds maximum quote")


def lower_func(x: float, sigma: float) -> tuple[float, float, float]:
    ax = abs(x)
    z = -ax / (sigma * SQRT3)
    z2 = z * z
    sigma2 = sigma * sigma
    cdf = float(NORMAL.cdf(z))  # type: ignore
    pdf = float(NORMAL.pdf(z))  # type: ignore
    norm2 = cdf * cdf
    term = 8.00 * SQRT3 * sigma * ax + (3 * sigma2 * (sigma2 - 8.00) - 8 * x * x) * (
        cdf / pdf
    )
    feval = 2 * math.pi * ax * cdf * norm2 / (3 * SQRT3)
    fprime = 2 * math.pi * z2 * norm2 * math.exp(z2 + sigma2 / 8.00)
    fpprime = (
        math.pi
        / 6
        * (z2 / (sigma2 * sigma))
        * cdf
        * math.exp(2 * z2 + sigma2 / 4.00)
        * term
    )

    return feval, fprime, fpprime


def rational(
    xl: float, xr: float, fl: float, fr: float, sl: float, sr: float, r: float
):
    def out(x: float) -> float:
        h = xr - xl
        s = (x - xl) / h
        s2 = s * s
        s1 = 1 - s
        s12 = s1 * s1
        denom = (
            fr * s2 * s
            + (r * fr - h * sr) * s2 * s1
            + (r * fl + h * sl) * s * s12
            + fl * s1 * s12
        )
        num = 1 + (r - 3) * s * s1
        return denom / num

    return out


def rational_auto_right(
    xl: float, xr: float, fl: float, fr: float, sl: float, sr: float, ssr: float
):
    r = r_auto_right(xr - xl, fl, fr, sl, sr, ssr)

    def out(x: float) -> float:
        h = xr - xl
        s = (x - xl) / h
        s2 = s * s
        s1 = 1 - s
        s12 = s1 * s1
        denom = (
            fr * s2 * s
            + (r * fr - h * sr) * s2 * s1
            + (r * fl + h * sl) * s * s12
            + fl * s1 * s12
        )
        num = 1 + (r - 3) * s * s1
        return denom / num

    return out


def rational_auto_left(
    xl: float, xr: float, fl: float, fr: float, sl: float, sr: float, ssl: float
):
    r = r_auto_left(xr - xl, fl, fr, sl, sr, ssl)

    def out(x: float) -> float:
        h = xr - xl
        s = (x - xl) / h
        s2 = s * s
        s1 = 1 - s
        s12 = s1 * s1
        denom = (
            fr * s2 * s
            + (r * fr - h * sr) * s2 * s1
            + (r * fl + h * sl) * s * s12
            + fl * s1 * s12
        )
        num = 1 + (r - 3) * s * s1
        return denom / num

    return out


def r_auto_left(
    dx: float, fl: float, fr: float, sl: float, sr: float, ssl: float
) -> float:
    d = (fr - fl) / dx
    return (0.5 * dx * ssl + (sr - sl)) / (d - sl)


def r_auto_right(
    dx: float, fl: float, fr: float, sl: float, sr: float, ssr: float
) -> float:
    d = (fr - fl) / dx
    return (0.5 * dx * ssr + (sr - sl)) / (sr - d)


def iota_parity(x: float, is_call: bool) -> float:
    sgn = -1.00
    if is_call:
        sgn = 1.00
    return np.max(np.exp(sgn * x / 2) - np.exp(-sgn * x / 2), 0)
